package FunCLBMSpark

import KMeansSpark.KMeans
import breeze.linalg.{DenseMatrix, DenseVector, max, min}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.stat.distribution.MultivariateGaussian
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import Tools._

import scala.util.Random

object Initialization  {

  def initialize(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                 condLatentBlock: CondLatentBlock,
                 EMMethod: String,
                 n:Int, p :Int,
                 verbose:Boolean = true,
                 initMethod: String = "randomPartition")(implicit ss: SparkSession): (CondLatentBlockModel, List[Int]) = {

    val KVec = condLatentBlock.getKVec
    val nSampleForLBMInit = min(n, 50)
    initMethod match {
      case "random" => {
        val model = Initialization.initFromComponentSample(periodogram, KVec, nSampleForLBMInit,verbose)
        (model, (0 until p).map(j => sample(model.proportionsCols)).toList)
      }
      case "randomPartition" => {
        initFromRandomPartition(periodogram, KVec, n,p,verbose)
      }
      case "KMeans" => {
        Initialization.initFromColKMeans(periodogram,KVec,n,verbose)
      }
      case "FunLBM" => {
        Initialization.initFromFunLBM(periodogram,KVec,n,verbose)
      }
      case _ => {
        println(s"Warning: No initial method has been provided and initMethod $initMethod provided " +
          "does not match possible initialization method name (\"random\",\"randomPartition\",\"KMeans\",\"FunLBM\")" +
          "Continuing with random initialization..")
        val model = Initialization.initFromComponentSample(periodogram,KVec,nSampleForLBMInit)
        (model, (0 until p).map(j => sample(model.proportionsCols)).toList)
      }
    }
  }


  def initFromComponentSample(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                              KVec: List[Int],
                              nSamples:Int = 10,
                              verbose: Boolean=false)(implicit ss: SparkSession): CondLatentBlockModel = {

    if(verbose) println("Random Sample Initialization")

    // %%%%%%%%%%%%%%%%%%%%%%
    val flattenedPeriodogram: RDD[DenseVector[Double]] = periodogram.map(_._2).flatMap(row => row.toArray.toList)
    val flattenedRDDAsList : RDD[(Int, Int, List[Double])] = flattenedPeriodogram.map(r => (0,0, r.toArray.toList))
    val (unsortedPcaCoefs, loadings) = TSSInterface.getPcaAndLoadings(flattenedRDDAsList)
    val pcaCoefs: RDD[(Int, Array[DenseVector[Double]])] = periodogram.map(row => (row._1, row._2.map(e => loadings * e)))
    // %%%%%%%%%%%%%%%%%%%%%%

    val L = KVec.length
    val MultivariateGaussians: List[List[MultivariateGaussian]] = KVec.indices.map(l => {
      (0 until KVec(l)).map(k_l => {
        val sampleBlock: List[DenseVector[Double]] = pcaCoefs.takeSample(withReplacement = false, nSamples)
          .map(e => Random.shuffle(e._2.toList).head).toList
        val mode: DenseVector[Double] = mean(sampleBlock)
        new MultivariateGaussian(Vectors.dense(mode.toArray), denseMatrixToMatrix(
          covariance(sampleBlock, mode)))
      }).toList
    }).toList

    val rowProportions:List[List[Double]] = (0 until L).map(l => {List.fill(KVec(l))(1.0 / KVec(l))}).toList
    val colProportions:List[Double] =  List.fill(L)(1.0 / L):List[Double]
    val loadingsList = (0 until L).map(l => {List.fill(KVec(l))(loadings)}).toList
    FunCLBMSpark.CondLatentBlockModel(rowProportions, colProportions, loadingsList, MultivariateGaussians)
  }

  def initFromGivenPartition(periodogram: RDD[(Int, Array[DenseVector[Double]], List[Int])],
                             colPartition: List[Int],
                             KVec: List[Int],
                             n: Int)(implicit ss: SparkSession): (CondLatentBlockModel, List[Int]) = {

    val flattenedPeriodogram: RDD[DenseVector[Double]] = periodogram.map(_._2).flatMap(row => row.toArray.toList)
    val flattenedRDDAsList : RDD[(Int, Int, List[Double])] = flattenedPeriodogram.map(r => (0,0, r.toArray.toList))
    val (unsortedPcaCoefs, loadings) = TSSInterface.getPcaAndLoadings(flattenedRDDAsList)

    val dataAndSizeByBlock =  KVec.indices.map(l => {
      (0 until KVec(l)).map(k_l => {
        val filteredData = periodogram.filter(_._3(l) == k_l).map(row => {row._2.zipWithIndex.filter(s => colPartition(s._2) == l).map(_._1)})
        val sizeBlock: Int = filteredData.map(_.length).sum().toInt
        require(sizeBlock > 0, "Algorithm could not converge: empty block")
        (filteredData, sizeBlock)
      })
    })

    val modelsAndLoadings: List[List[(MultivariateGaussian, DenseMatrix[Double])]] =
      KVec.indices.map(l => {
        (0 until KVec(l)).map(k_l => {
          val filteredRDD = dataAndSizeByBlock(l)(k_l)._1
          val sizeBlock = dataAndSizeByBlock(l)(k_l)._2
          val pcaCoefs = filteredRDD.map(row => row.map(e => loadings * e))
          val mode: DenseVector[Double] = pcaCoefs.map(_.reduce(_ + _)).reduce(_ + _) / (sizeBlock - 1).toDouble
          val covariance: DenseMatrix[Double] = pcaCoefs.map(_.map(v => {
            val vc: DenseVector[Double] = v - mode
            vc * vc.t
          }).reduce(_ + _)).reduce(_ + _) / (sizeBlock - 1).toDouble
          val model: MultivariateGaussian = new MultivariateGaussian(Vectors.dense(mode.toArray), denseMatrixToMatrix(covariance))
          (model,loadings)
        }).toList
      }).toList

    val countRows: Map[(Int,Int), Int] = periodogram.map(row => {
      KVec.indices.map(l => {(row._3(l),l)}).toList
    }).reduce(_ ++ _).groupBy(identity).mapValues(_.size)
    val proportionRows: List[List[Double]] = KVec.indices.map(l => {
      (0 until KVec(l)).map(k_l => {countRows(k_l,l)/n.toDouble}).toList
    }).toList

    val countCols = colPartition.groupBy(identity).mapValues(_.size)
    val proportionCols = countCols.map(c => c._2 / countCols.values.sum.toDouble).toList
    val newModels = modelsAndLoadings.map(s => s.map(_._1))
    val newLoadings = modelsAndLoadings.map(s => s.map(_._2))

    (FunCLBMSpark.CondLatentBlockModel(proportionRows, proportionCols, newLoadings, newModels), colPartition)
  }

  def initFromRandomPartition(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                              KVec: List[Int],
                              n: Int,
                              p: Int,
                              verbose: Boolean=false)(implicit ss: SparkSession): (CondLatentBlockModel, List[Int]) = {
    if(verbose) println("Random Partition Initialization")


    val colPartition: List[Int] = Random.shuffle((0 until p).map(j => j%KVec.length)).toList
    val randomRowPartition: List[List[Int]] = KVec.indices.map(l => {
      Random.shuffle((0 until n).map(_%KVec(l))).toList
    }).toList

    val periodogramWithRowPartition = Tools.joinRowPartitionToData(periodogram, randomRowPartition,n )

    initFromGivenPartition(periodogramWithRowPartition, colPartition, KVec, n)
  }

  def initFromColKMeans(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                        KVec: List[Int],
                        n: Int,
                        verbose:Boolean=false)(implicit ss: SparkSession): (CondLatentBlockModel, List[Int]) = {


    if(verbose) println("KMeans Initialization")

    // %%%%%%%%%%%%%%%%%%%%%%
    val flattenedPeriodogram: RDD[DenseVector[Double]] = periodogram.map(_._2).flatMap(row => row.toArray.toList)
    val flattenedRDDAsList : RDD[(Int, Int, List[Double])] = flattenedPeriodogram.map(r => (0,0, r.toArray.toList))
    val (unsortedPcaCoefs, loadings) = TSSInterface.getPcaAndLoadings(flattenedRDDAsList)
    val pcaCoefs: RDD[(Int, Array[DenseVector[Double]])] = periodogram.map(row => (row._1, row._2.map(e => loadings * e)))
    // %%%%%%%%%%%%%%%%%%%%%%

    val L = KVec.length
    val p = pcaCoefs.take(1).head._2.length

    val dataByCol = Tools.inverseIndexedList(pcaCoefs.collect().toList)
    val dataByColRDD = ss.sparkContext.parallelize(dataByCol,200)
    val kmeanCol = new KMeans()
    val resModelCol = kmeanCol.setK(L).run(dataByColRDD, verbose=verbose)
    val colPartition = resModelCol("Partition").asInstanceOf[List[Int]]

    println(colPartition)
    val rowPartition = (0 until L).map(l => {
      val filteredDataByCol = dataByCol.filter(r => colPartition(r._1)==l)
      val filteredDataByRow = Tools.inverseIndexedList(filteredDataByCol)
      val filteredDataByRowRDD = ss.sparkContext.parallelize(filteredDataByRow,100)
      val kmeanRow = new KMeans().setK(KVec(l))
      val resModelRow = kmeanRow.run(filteredDataByRowRDD, verbose=verbose)
      println(resModelRow("Partition").asInstanceOf[List[Int]])
      resModelRow("Partition").asInstanceOf[List[Int]]
    }).toList

    val periodogramWithRowPartition = Tools.joinRowPartitionToData(periodogram, rowPartition, n)

    initFromGivenPartition(periodogramWithRowPartition, colPartition, KVec, n)
  }

  def initFromFunLBM(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                      KVec: List[Int],
                      n:Int,
                      verbose:Boolean = false)(implicit ss: SparkSession): (CondLatentBlockModel, List[Int]) = {
    if(verbose) println("FunLBM Initialization")

    // %%%%%%%%%%%%%%%%%%%%%%
    val flattenedPeriodogram: RDD[DenseVector[Double]] = periodogram.map(_._2).flatMap(row => row.toArray.toList)
    val flattenedRDDAsList : RDD[(Int, Int, List[Double])] = flattenedPeriodogram.map(r => (0,0, r.toArray.toList))
    val (unsortedPcaCoefs, loadings) = TSSInterface.getPcaAndLoadings(flattenedRDDAsList)
    val pcaCoefs: RDD[(Int, Array[DenseVector[Double]])] = periodogram.map(row => (row._1, row._2.map(e => loadings * e)))
    // %%%%%%%%%%%%%%%%%%%%%%

    val L = KVec.length
    val maxK = max(KVec)
    val dataByCol = Tools.inverseIndexedList(pcaCoefs.collect().toList)

    val latentBlock = new FunLBMSpark.LatentBlock()
    latentBlock.setMaxIterations(5).setMaxBurninIterations(5).setL(L).setK(maxK).setUpdateLoadingStrategy(4)
    val resLBM = latentBlock.run(periodogram, verbose= true,
    nConcurrent = 1, nTryMaxPerConcurrent = 10)
    val colPartition: List[Int] = resLBM("ColPartition").asInstanceOf[List[Int]]

    val rowPartition = (0 until L).map(l => {
      val filteredDataByCol = dataByCol.filter(r => colPartition(r._1)==l)
      val filteredDataByRow = Tools.inverseIndexedList(filteredDataByCol)
      val filteredDataByRowRDD = ss.sparkContext.parallelize(filteredDataByRow,100)
      val latentBlock = new FunCLBMSpark.CondLatentBlock().setKVec(List(KVec(l))).setUpdateLoadingStrategy(4)
      val resModelRow = latentBlock.run(filteredDataByRowRDD, verbose=true, initMethod ="random",
        nConcurrent = 1, nTryMaxPerConcurrent = 10)
      val rowPartition = resModelRow("RowPartition").asInstanceOf[List[List[Int]]].head
      rowPartition
    }).toList
    val periodogramWithRowPartition = Tools.joinRowPartitionToData(periodogram, rowPartition, n)
    initFromGivenPartition(periodogramWithRowPartition, colPartition, KVec, n)
  }
}

