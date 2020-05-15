package FunCLBMSpark

import breeze.linalg.{*, DenseMatrix, DenseVector, argmax, sum}
import breeze.numerics.{exp, log}
import org.apache.spark.ml.stat.distribution.MultivariateGaussian
import org.apache.spark.mllib.clustering.GaussianMixtureModel
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import Tools._

case class CondLatentBlockModel(proportionsRows: List[List[Double]],
                                proportionsCols: List[Double],
                                loadings: List[List[DenseMatrix[Double]]],
                                gaussians: List[List[MultivariateGaussian]]) {
  val precision = 1e-5
  def KVec: List[Int] = gaussians.map(_.length)

  def this() {
    this(List(List(0D)),List(0D),
      List(List(DenseMatrix(0D))),
      List(List(new MultivariateGaussian(
      Vectors.dense(Array(0D)),
      denseMatrixToMatrix(DenseMatrix(1D))))))
  }
  // Version to call inside the SEM algorithm
  def StochasticExpectationStep(data: RDD[(Int, Array[DenseVector[Double]], List[Int])],
                                colPartition: List[Int],
                                p: Int,
                                nIter: Int = 3,
                                verbose: Boolean = true): (RDD[(Int, Array[DenseVector[Double]], List[Int])], List[Int]) = {

    var newData: RDD[(Int, Array[DenseVector[Double]], List[Int])] = drawRowPartition(data, colPartition)
    var newColPartition: List[Int] = drawColPartition(newData)

    var k: Int = 1
    while (k < nIter) {
      newData = drawRowPartition(newData, newColPartition)
      newColPartition = drawColPartition(newData)
      k += 1
    }
    (newData, newColPartition)
  }

  def drawRowPartition(data: RDD[(Int, Array[DenseVector[Double]], List[Int])],
                       colPartition: List[Int]): RDD[(Int, Array[DenseVector[Double]], List[Int])] = {

    val jointLogDistribRows = computeJointLogDistribRowsFromSample(data, colPartition)

    jointLogDistribRows.map(x => {
      (x._1,
        x._2,
        x._3.indices.map(l => {
          val vecProb: List[Double] = x._3(l)
          val LSE = logSumExp(vecProb)
          val normalizedVecProb = vecProb.map(e => exp(e - LSE))
          val samp = sample(normalizedVecProb)
          samp
        }).toList)
    })
  }

  def computeJointLogDistribRowsFromSample(data: RDD[(Int, Array[DenseVector[Double]], List[Int])],
                                           colPartition: List[Int]): RDD[(Int, Array[DenseVector[Double]], List[List[Double]])] = {

    val logCondPiRows: List[List[Double]] = this.proportionsRows.map(piRow=> piRow.map(log(_)))
    val rangeCols: List[List[Int]] = KVec.indices.map(l => colPartition.zipWithIndex.filter(_._1 == l).map(_._2)).toList

    data.map(row => {
      (row._1,
        row._2,
        rangeCols.indices.map(l => {
          (0 until KVec(l)).map(k_l => {
            rangeCols(l).map(col => {
              this.gaussians(l)(k_l).logpdf(Vectors.dense((loadings(l)(k_l) * row._2(col)).toArray))
            }).sum + logCondPiRows(l)(k_l)
          }).toList
        }).toList)
    })
  }

  def drawColPartition(data: RDD[(Int, Array[DenseVector[Double]], List[Int])]): List[Int] = {

    val jointLogDistribCols: List[List[Double]] = computeJointLogDistribColsFromSample(data)
    val res = jointLogDistribCols.map(x => {
      {
        val LSE = logSumExp(x)
        sample(x.map(e => exp(e - LSE)))
      }
    })
    res
  }

  def computeJointLogDistribColsFromSample(data: RDD[(Int, Array[DenseVector[Double]], List[Int])]): List[List[Double]] = {

    val logPiCols: DenseVector[Double] = DenseVector(this.proportionsCols.map(log(_)).toArray)

    val D: RDD[DenseMatrix[Double]] =
      data.map(row => {
        row._3.indices.map(l => {
          DenseMatrix(
            row._2.indices.map(j => {
              this.gaussians(l)(row._3(l))
                .logpdf(Vectors.dense((loadings(l)(row._3(l))*row._2(j)).toArray))
            }).toArray)

        }).reduce(DenseMatrix.vertcat(_,_))
      })

    val prob = D.reduce(_ + _).t
    require(prob.cols == logPiCols.length,  "Algorithm could not converge: empty col cluster drawn in SE Step")
    val sumProb = prob(*, ::).map(dv => dv.toArray.toList).toArray.toList.zipWithIndex.map(e => {
      (DenseVector(e._1.toArray) + logPiCols).toArray.toList
    })

    sumProb
  }

  def SEMGibbsMaximizationStep(periodogram: RDD[(Int, Array[DenseVector[Double]], List[Int])],
                               colPartition: List[Int],
                               n:Int,
                               verbose: Boolean = true,
                               updateLoadings: Boolean = true)(implicit ss: SparkSession): CondLatentBlockModel = {
    periodogram.cache()
    var t0 = System.nanoTime()

//    if(verbose){println("SizeBlock computation")}

    // Data is filtered in a dedicated loop in order to detect empty blocks before any pca is run
    val dataAndSizeByBlock =  KVec.indices.map(l => {
      (0 until KVec(l)).map(k_l => {
        val filteredData = periodogram.filter(_._3(l) == k_l).map(row => {
          row._2.zipWithIndex.filter(s => colPartition(s._2) == l).map(_._1)
        })
        val sizeBlock: Int = filteredData.map(_.length).sum().toInt
//        if(verbose){println("block ("+k_l.toString+", "+l.toString+") size: "+sizeBlock.toString+", ")}
        require(sizeBlock > 0, "Algorithm could not converge: empty block")
        (filteredData, sizeBlock)
      })
    })

    val newLoadings = if(updateLoadings){
      KVec.indices.map(l => {
        (0 until KVec(l)).map(k_l => {
          val filteredRDD = dataAndSizeByBlock(l)(k_l)._1
          val flattenedPeriodogram: RDD[DenseVector[Double]] = filteredRDD.flatMap(row => row)
          val flattenedRDDAsList : RDD[(Int, Int, List[Double])] = flattenedPeriodogram.map(r => (0,0, r.toArray.toList))
          val (unsortedPcaCoefs, loadingBlock) = TSSInterface.getPcaAndLoadings(flattenedRDDAsList)
          loadingBlock
        }).toList
      }).toList
    } else {this.loadings}

    val newModels: List[List[MultivariateGaussian]] =
      KVec.indices.map(l => {
        (0 until KVec(l)).map(k_l => {

          val filteredRDD = dataAndSizeByBlock(l)(k_l)._1
          val sizeBlock = dataAndSizeByBlock(l)(k_l)._2
          val pcaCoefs = filteredRDD.map(row => row.map(e => newLoadings(l)(k_l) * e))

          val mode: DenseVector[Double] = pcaCoefs.map(_.reduce(_ + _)).reduce(_ + _) / (sizeBlock - 1).toDouble
          val covariance: DenseMatrix[Double] = pcaCoefs.map(_.map(v => {
            val vc: DenseVector[Double] = v - mode
            vc * vc.t
          }).reduce(_ + _)).reduce(_ + _) / (sizeBlock - 1).toDouble
          val model: MultivariateGaussian = new MultivariateGaussian(Vectors.dense(mode.toArray), denseMatrixToMatrix(covariance))
//          if(verbose){t0 = LBM.Tools.printTime(t0, "> block ("+k_l.toString+", "+l.toString+") - size: "+ sizeBlock.toString)}
          model
        }).toList
      }).toList

    val countRows: Map[(Int,Int), Int] = periodogram.map(row => {
      KVec.indices.map(l => {
        (row._3(l),l)
      }).toList
    }).reduce(_ ++ _).groupBy(identity).mapValues(_.size)
    val newProportionRows: List[List[Double]] = KVec.indices.map(l => {
      (0 until KVec(l)).map(k_l => {
        countRows(k_l,l)/n.toDouble
      }).toList
    }).toList

    val countCols = colPartition.groupBy(identity).mapValues(_.size)
    val newProportionCols = countCols.map(c => c._2 / countCols.values.sum.toDouble).toList

    CondLatentBlockModel(newProportionRows, newProportionCols, newLoadings, newModels)
  }

  def completelogLikelihood(periodogram: RDD[(Int, Array[DenseVector[Double]], List[Int])],
                            colPartition: List[Int]): Double = {

    val logRho: List[Double] = proportionsCols.map(log(_))
    val logPi: List[List[Double]]  = proportionsRows.map(_.map(log(_)))
    val rangeCols: List[List[Int]] = KVec.indices.map(l => colPartition.zipWithIndex.filter(_._1 == l).map(_._2)).toList

    periodogram.map(row => {
      rangeCols.indices.map(l => {
        rangeCols(l).map(j => {
          logPi(l)(row._3(l))
          + logRho(l)
          + this.gaussians(l)(row._3(l)).logpdf(Vectors.dense((loadings(l)(row._3(l)) *row._2(j)).toArray))
        }).sum
      }).sum
    }).sum()
  }

  def ICL(completelogLikelihood: Double,
          n: Double,
          p: Double,
          fullCovariance: Boolean): Double = {

    val dimVar = this.gaussians.head.head.mean.size
    val L = KVec.length
    val nParamPerComponent = if(fullCovariance){
      dimVar+ dimVar*(dimVar+1)/2D
    } else {2D * dimVar}

    val nClusterRow = this.KVec.sum

    (completelogLikelihood
      - log(n)*(nClusterRow - L)/2D
      - log(p)*(L-1)/2D
      - log(n*p)*(nClusterRow*L*nParamPerComponent)/2D)
  }

}

