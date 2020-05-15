
import FunCLBMSpark.DataGeneration._
import FunCLBMSpark.{OutputResults, TSSInterface, Tools}
import breeze.linalg.{DenseMatrix, DenseVector}
import com.github.unsupervise.spark.tss.core.TSS
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import smile.validation.adjustedRandIndex

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
  val confSpark = new SparkConf().setMaster("local[2]").setAppName("LBM")
  val prototypes: List[List[(List[Double], Double) => List[Double]]] = List(List(f1, f2, f3), List(f4, f5), List(f6,f7))
  val sigma = 0.2
  val sizeClusterRow = List(List(20, 40, 30), List(60, 30), List(40, 50))
  val sizeClusterCol = List(45, 15, 30)

  val tss: TSS = randomDataGeneration(prototypes, sigma, sizeClusterRow, sizeClusterCol, shuffled=false)
  val periodogram = TSSInterface.getPeriodograms(tss)
    .map(row => (row._1, (row._2, row._3)))
  val dataRDDByRow: RDD[(Int, Array[DenseVector[Double]])] = periodogram.groupByKey.map(row => (row._1,
    row._2.toList.sortBy(_._1).map(r => DenseVector(r._2.toArray)).toArray))

  val n: Int = sizeClusterRow.head.sum
  val p: Int = sizeClusterCol.sum
  val KVec = List(3, 2, 2)

  val trueRowPartition = sizeClusterRow.indices.map(l => sizeClusterRow(l).indices.map(i =>
    List.fill(sizeClusterRow(l)(i)){i}).reduce(_++_)).toList
  val trueEntireRowPartition = Tools.getEntireRowPartition(trueRowPartition)
  val trueColPartition = sizeClusterCol.indices.map(j => List.fill(sizeClusterCol(j)){j}).reduce(_++_)
  val trueBlockPartition = Tools.getBlockPartition(trueRowPartition, trueColPartition)

  def main(args: Array[String]) {

    println("Generate a bunch of results ARI / ICL / Likelihood")
    var t0 = System.nanoTime()
    (0 until 1).foreach(i => {
      println(">>>>> " + i.toString)

      val latentBlock = new FunCLBMSpark.CondLatentBlock()
        .setKVec(KVec).setMaxIterations(3).setMaxBurninIterations(3).setUpdateLoadingStrategy(3)
      val outputSEM = latentBlock.run(dataRDDByRow, nConcurrent=1, nTryMaxPerConcurrent = 10,
        initMethod = "randomPartition", verbose = true)

      val icl = outputSEM("ICL").asInstanceOf[List[Double]].last
      val ll = outputSEM("LogLikelihood").asInstanceOf[List[Double]].last
      val rowPartition = outputSEM("RowPartition").asInstanceOf[List[List[Int]]]
      val colPartition = outputSEM("ColPartition").asInstanceOf[List[Int]]
      val estimatedBlockPartition = Tools.getBlockPartition(rowPartition, colPartition)
      val estimatedEntireRowPartition = Tools.getEntireRowPartition(rowPartition)
      val scoresMat = DenseMatrix(Array(
        adjustedRandIndex(trueEntireRowPartition, estimatedEntireRowPartition),
        adjustedRandIndex(trueColPartition.toArray, colPartition.toArray),
        adjustedRandIndex(trueBlockPartition, estimatedBlockPartition),
        ll, icl).map(_.toString)).reshape(1, 5)
      t0 = Tools.printTime(t0,"Model Selection ")

      OutputResults.writeMatrixStringToCsv("src/main/results/result.csv",
        scoresMat, append = true)
    })
  }
}
