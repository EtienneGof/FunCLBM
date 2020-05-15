package FunLBMSpark

import FunCLBMSpark.Tools._
import breeze.linalg.{DenseVector, min}
import breeze.stats.mean
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.stat.distribution.MultivariateGaussian
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.util.Random

object Initialization  {

  def initialize(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                 condLatentBlock: LatentBlock,
                 EMMethod: String,
                 n:Int, p :Int,
                 verbose:Boolean = true,
                 initMethod: String = "random")(implicit ss: SparkSession): (LatentBlockModel, List[Int]) = {

    val K = condLatentBlock.getK
    val L = condLatentBlock.getL

    val nSampleForLBMInit = min(n, 50)
    initMethod match {
      case "random" => {
        val model = Initialization.randomModelInitialization(periodogram, K, L, nSampleForLBMInit)
        (model, (0 until p).map(j => sample(model.proportionsCols)).toList)
      }
//
      case _ => {
        println(s"Warning: No initial method has been provided and initMethod $initMethod provided " +
          "does not match possible initialization method name (\"random\",\"sample\")" +
          "Continuing with random initialization..")
        val model = Initialization.randomModelInitialization(periodogram, K, L, nSampleForLBMInit)
        (model, (0 until p).map(j => sample(model.proportionsCols)).toList)
      }
    }
  }

  def randomModelInitialization(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                                K: Int, L:Int,
                                nSamples:Int = 10)(implicit ss: SparkSession): LatentBlockModel = {

    // %%%%%%%%%%%%%%%%%%%%%%
    val flattenedPeriodogram: RDD[DenseVector[Double]] = periodogram.map(_._2).flatMap(row => row.toArray.toList)
    val flattenedRDDAsList : RDD[(Int, Int, List[Double])] = flattenedPeriodogram.map(r => (0,0, r.toArray.toList))
    val (unsortedPcaCoefs, loadings) = FunCLBMSpark.TSSInterface.getPcaAndLoadings(flattenedRDDAsList)
    val pcaCoefs: RDD[(Int, Array[DenseVector[Double]])] = periodogram.map(row => (row._1, row._2.map(e => loadings * e)))
    // %%%%%%%%%%%%%%%%%%%%%%

    val MultivariateGaussians: List[List[MultivariateGaussian]] = (0 until K).map(k => {
      (0 until L).map(l => {
        val sampleBlock: List[DenseVector[Double]] = pcaCoefs.takeSample(false, nSamples)
          .map(e => Random.shuffle(e._2.toList).head).toList
        val mode: DenseVector[Double] = FunCLBMSpark.Tools.mean(sampleBlock)
        new MultivariateGaussian(Vectors.dense(mode.toArray), denseMatrixToMatrix(covariance(sampleBlock, mode)))
      }).toList
    }).toList

    val rowProportions:List[Double] =  List.fill(K)(1.0 / K):List[Double]
    val colProportions:List[Double] =  List.fill(L)(1.0 / L):List[Double]
    val loadingsList =  List.fill(K){List.fill(L)(loadings)}

    LatentBlockModel(rowProportions, colProportions, loadingsList, MultivariateGaussians)
  }

//  def initFromFunCLBMOnSample(data: RDD[(Int, Array[DenseVector[Double]])],
//                                KVec: List[Int],
//                                proportionSample:Double = 0.2,
//                                nTry: Int = 5,
//                                nConcurrentPerTry: Int = 3)(implicit ss: SparkSession): (CondLatentBlockModel, List[Int]) = {
//
//    require(proportionSample>0 & proportionSample<=1, "proportionSample argument is a proportion (should be >0 and <=1)")
//
//    val resList = (0 until nTry).map(i => {
//      val dataSample = data.sample(withReplacement = false, proportionSample)
//      val CLBM = new CondLatentBlock().setKVec(KVec)
//      CLBM.run(dataSample, nConcurrent=nConcurrentPerTry, nTryMaxPerConcurrent=10,initMethod = "random")
//    })
//
//    val allLikelihoods: DenseVector[Double] = DenseVector(resList.map(_("LogLikelihood").asInstanceOf[List[Double]].last).toArray)
//    val bestRes = resList(argmax(allLikelihoods))
//
//    (bestRes("Model").asInstanceOf[CondLatentBlockModel], bestRes("ColPartition").asInstanceOf[List[Int]])
//  }

}

