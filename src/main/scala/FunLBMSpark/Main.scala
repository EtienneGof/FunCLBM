package FunLBMSpark

import FunCLBMSpark.DataGeneration._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession


object Main {

  implicit val ss = SparkSession
    .builder()
    .master("local[*]")
    .appName("AnalysePlan")
    .config("spark.executor.cores", 2)
    //.config("spark.executor.memory", "30G")
    .config("spark.executor.heartbeatInterval", "20s")
    .config("spark.driver.memory", "10G")
    .getOrCreate()

  ss.sparkContext.setLogLevel("WARN")
  ss.sparkContext.setCheckpointDir("checkpointDir")

  val KVec = List(3, 2)
  val L = 2
  val confSpark = new SparkConf().setMaster("local[2]").setAppName("LBM")

  val prototypes: List[List[(List[Double],Double)=> List[Double]]] = List(List(f1, f2, f3), List(f4, f5))
  val sigma = 0.001

  val sizeClusterRow = List(List(20, 20, 20), List(30, 30))
  val sizeClusterCol = List(20, 20)

  def main(args: Array[String]) {
//
//    val tss: TSS = randomDataGeneration(prototypes,sigma,sizeClusterRow,sizeClusterCol)

//    val (pcaCoefs, frequencies, scaler, loadings):(RDD[Row], Array[Double], StandardScalerModel, org.apache.spark.ml.linalg.DenseMatrix) =
//      TSSInterface.getPCACoef(tss)
//
//    val dataRDDWithIntIndex: RDD[(Int, (Int, BzDenseVector[Double]))]= pcaCoefs.map(row => (
//      row(1).asInstanceOf[String].toInt, // iterationId as Index
//      (row(0).asInstanceOf[String].toInt,    // varName as Index
//        BzDenseVector(row(2).asInstanceOf[Seq[Double]].toArray))) // Fourier coefs
//    )
//
//    val dataRDDByRow: RDD[(Int, Array[BzDenseVector[Double]])] = dataRDDWithIntIndex.groupByKey.map(row => (row._1, row._2.toList.sortBy(_._1).map(_._2).toArray))
//
//    val latentBlock = new CLBMSpark.CondLatentBlock()
//    latentBlock.setKVec(KVec).setMaxIterations(6).setMaxBurninIterations(6)
//
//    val outputSEM = latentBlock.run(dataRDDByRow, sc = ss.sparkContext, nConcurrent=5, initMethod = "random", verbose = true)
//    val resModel = outputSEM("Model").asInstanceOf[CondLatentBlockModel]
//
//    val colPartition = outputSEM("ColPartition").asInstanceOf[List[Int]]
//    val rowPartition = outputSEM("RowPartition").asInstanceOf[List[List[Int]]]
//    val logLikelihood = outputSEM("LogLikelihood").asInstanceOf[List[List[Int]]]

  }

}
