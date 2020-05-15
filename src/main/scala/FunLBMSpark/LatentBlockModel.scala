package FunLBMSpark

import FunCLBMSpark.TSSInterface
import FunCLBMSpark.Tools._
import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.numerics.{exp, log}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.stat.distribution.MultivariateGaussian
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

case class LatentBlockModel(proportionsRows: List[Double],
                            proportionsCols: List[Double],
                            loadings: List[List[DenseMatrix[Double]]],
                            gaussians: List[List[MultivariateGaussian]]) {
  val precision = 1e-8
  val K = proportionsRows.length
  val L = proportionsCols.length

  // Auxiliary constructor also takes a String?? (compile error)
  def this() {
    this(
      List(0D),
      List(0D),
      List(List(DenseMatrix(0D))),
      List(List(new MultivariateGaussian(
      Vectors.dense(Array(0D)),
      denseMatrixToMatrix(DenseMatrix(0D))))))
  }

  // Version to call inside the SEM algorithm
  def StochasticExpectationStep(data: RDD[(Int, Array[DenseVector[Double]], Int)],
                                colPartition: List[Int],
                                p: Int,
                                nIter: Int = 3,
                                verbose: Boolean = true)= {

    var newData: RDD[(Int, Array[DenseVector[Double]], Int)] = drawRowPartition(data, colPartition)
    var newColPartition: List[Int] = drawColPartition(newData)

    var k: Int = 1
    while (k < nIter) {
      newData = drawRowPartition(newData, newColPartition)
      newColPartition = drawColPartition(newData)
      k += 1
    }
    (newData, newColPartition)
  }

  def drawRowPartition(data: RDD[(Int, Array[DenseVector[Double]], Int)],
                       colPartition: List[Int]): RDD[(Int, Array[DenseVector[Double]], Int)] = {

    val jointLogDistribRows = computeJointLogDistribRowsFromSample(data, colPartition)
    jointLogDistribRows.map(x => {
      (x._1,
        x._2, {
        val LSE = logSumExp(x._3)
        sample(x._3.map(e => exp(e - LSE)))
      })
    })
  }

  def computeJointLogDistribRowsFromSample(data: RDD[(Int, Array[DenseVector[Double]], Int)],
                                           colPartition: List[Int]): RDD[(Int, Array[DenseVector[Double]], List[Double])] = {

    val logPiRows: DenseVector[Double] = DenseVector(this.proportionsRows.map(log(_)).toArray)

    data.map(row => {
      (row._1,
        row._2,
        (0 until this.K).map(k => {
          row._2.indices.map(j => {
            this.gaussians(k)(colPartition(j)).logpdf(Vectors.dense((loadings(k)(colPartition(j))*row._2(j)).toArray))
          }).sum + logPiRows(k)
        }).toList)
    })
  }


  def drawColPartition(data: RDD[(Int, Array[DenseVector[Double]], Int)]): List[Int] = {

    val jointLogDistribCols: List[List[Double]] = computeJointLogDistribColsFromSample(data)
    val res = jointLogDistribCols.map(x => {
      {
        val LSE = logSumExp(x)
        sample(x.map(e => exp(e - LSE)))
      }
    })
    res
  }

  def computeJointLogDistribColsFromSample(data: RDD[(Int, Array[DenseVector[Double]], Int)]): List[List[Double]] = {

    val logPiCols: DenseVector[Double] = DenseVector(this.proportionsCols.map(log(_)).toArray)
    val D: RDD[DenseMatrix[Double]] =
      data.map(row => {
        row._2.indices.map(j => {
          DenseMatrix((0 until this.L).map(l => {
            this.gaussians(row._3)(l)
              .logpdf(Vectors.dense((loadings(row._3)(l)*row._2(j)).toArray))
          }).toArray)
        }).reduce((a, b) => DenseMatrix.vertcat(a, b))
      })

    val prob = D.reduce(_ + _)
    val sumProb = prob(*, ::).map(dv => dv.toArray.toList).toArray.toList.zipWithIndex.map(e =>
      (DenseVector(e._1.toArray) + logPiCols).toArray.toList)

    sumProb
  }

  def SEMGibbsMaximizationStep(periodogram: RDD[(Int, Array[DenseVector[Double]], Int)],
                               colPartition: List[Int],
                               n:Int,
                               verbose: Boolean = true,
                               updateLoadings: Boolean = true)(implicit ss: SparkSession): LatentBlockModel = {
    periodogram.cache()

    val dataAndSizeByBlock = (0 until this.K).map(k => {
      (0 until this.L).map(l => {
        val filteredData = periodogram.filter(_._3 == k).map(row => {
          row._2.zipWithIndex.filter(s => colPartition(s._2) == l).map(_._1)
        })
        val sizeBlock: Int = filteredData.map(_.length).sum().toInt
        //        if(verbose){println("block ("+k_l.toString+", "+l.toString+") size: "+sizeBlock.toString+", ")}
        require(sizeBlock > 0, "Algorithm could not converge: empty block")
        (filteredData, sizeBlock)
      })
    })

    val newLoadings: List[List[DenseMatrix[Double]]] = if(updateLoadings){
      (0 until this.K).map(k => {
        (0 until this.L).map(l => {
          val filteredRDD = dataAndSizeByBlock(k)(l)._1
          val flattenedPeriodogram: RDD[DenseVector[Double]] = filteredRDD.flatMap(row => row)
          val flattenedRDDAsList : RDD[(Int, Int, List[Double])] = flattenedPeriodogram.map(r => (0,0, r.toArray.toList))
          val (unsortedPcaCoefs, loadingBlock) = TSSInterface.getPcaAndLoadings(flattenedRDDAsList)
          loadingBlock
        }).toList
      }).toList
    } else {this.loadings}

    val newModels: List[List[MultivariateGaussian]] =
      (0 until this.K).map(k => {
        (0 until this.L).map(l => {

          val filteredRDD = dataAndSizeByBlock(k)(l)._1
          val sizeBlock = dataAndSizeByBlock(k)(l)._2
          val pcaCoefs = filteredRDD.map(row => row.map(e => newLoadings(k)(l) * e))

          val mode: DenseVector[Double] = pcaCoefs.map(_.reduce(_ + _)).reduce(_ + _) / (sizeBlock - 1).toDouble
          val covariance: DenseMatrix[Double] = pcaCoefs.map(_.map(v => {
            val vc: DenseVector[Double] = v - mode
            val vcInternProd: DenseMatrix[Double] = vc * vc.t
            vcInternProd
          }).reduce(_ + _)).reduce(_ + _) / (sizeBlock - 1).toDouble
          val model: MultivariateGaussian = new MultivariateGaussian(Vectors.dense(mode.toArray), denseMatrixToMatrix(covariance))
          model
        }).toList
      }).toList

    val countRows: Map[Int, Int] = periodogram.map(row => {
      (row._3, 1)
    }).reduceByKey(_ + _).collect().toList.sortBy(_._1).toMap
    val proportionRows = countRows.map(c => c._2 / countRows.values.sum.toDouble).toList
    val countCols = colPartition.groupBy(identity).mapValues(_.size)
    val proportionCols = countCols.map(c => c._2 / countCols.values.sum.toDouble).toList

    LatentBlockModel(proportionRows, proportionCols, newLoadings, newModels)
  }

  def completelogLikelihood(data: RDD[(Int, Array[DenseVector[Double]], Int)],
                            colPartition: List[Int]): Double = {

    val logRho: List[Double] = this.proportionsCols.map(log(_))
    val logPi: List[Double]  = this.proportionsRows.map(log(_))

    data.map(row => {
      row._2.indices.map(j => {
        val l = colPartition(j)
        logPi(row._3)
        + logRho(l)
        + this.gaussians(row._3)(colPartition(j)).logpdf(Vectors.dense((loadings(row._3)(l)*row._2(j)).toArray))
      }).sum
    }).sum
  }

  def ICL(completelogLikelihood: Double,
          n: Double,
          p: Double,
          fullCovariance: Boolean): Double = {

    val dimVar = this.gaussians.head.head.mean.size
    val nParamPerComponent = if(fullCovariance){
      dimVar+ dimVar*(dimVar+1)/2D
    } else {2D * dimVar}
    completelogLikelihood - log(n)*(this.K - 1)/2D - log(p)*(L-1)/2D - log(n*p)*(K*L*nParamPerComponent)/2D
  }

}

