package FunCLBMSpark

import java.io.{BufferedWriter, File, FileWriter}


import org.apache.spark.sql.Row
import Tools._
import breeze.stats.distributions.MultivariateGaussian
import org.apache.spark.{SparkContext, sql}
import org.apache.spark.ml.linalg.DenseMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession, functions}
import org.apache.spark.sql.functions._
import com.github.unsupervise.spark.tss.core.TSS
import com.github.unsupervise.spark.tss.{functions => tssFunctions}
import org.apache.spark
import org.apache.spark.ml.feature.{PCA, StandardScaler, StandardScalerModel}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.linalg.Vectors
import breeze.linalg.{*, DenseMatrix => BzDenseMatrix, DenseVector => BzDenseVector}
import org.apache.commons.math3.analysis.function.Identity

import scala.collection.mutable
import scala.util.Random

object TSSInterface  {

  def toTSS(data: RDD[(Int, Int, List[Double], List[Double])])(implicit ss: SparkSession): TSS = {

    val rddNewEncoding = data.map(row =>
      (row._1.toString,
        row._2.toString,
        row._3.head,
        row._3.last,
        row._3(1)- row._3.head,
        row._4))

    val dfWithSchema = ss.createDataFrame(rddNewEncoding)
      .toDF("scenario_id", "varName", "timeFrom", "timeTo","timeGranularity", "series")

    val dfWithDecorator = dfWithSchema.select(
        map(lit("scenario_id"),
          col("scenario_id"),
          lit("varName"),
          col("varName")).alias("decorators"),
        col("timeFrom"),
        col("timeTo"),
        col("timeGranularity"),
        col("series").alias("series")
      )

    TSS(dfWithDecorator)
  }

  def addPCA(series: sql.DataFrame,
             outColName: String,
             inColName: String,
             maxK: Int,
             significancyThreshold: Double,
             pcaLoadingsOutFile: Option[File] = None,
             pcaVariancesOutFile: Option[File] = None,
             pcaCos2OutFile: Option[File] = None,
             pcaContribVariablesOutFile: Option[File] = None):(TSS, DenseMatrix) = {
    val pca = new PCA()
      .setInputCol(inColName)
      .setOutputCol(outColName)
      .setK(maxK)
      .fit(series)
    val significantPCADimensions =
      pca.explainedVariance.values.indices.find(i => {
        pca.explainedVariance.values.slice(0, i + 1).sum >= significancyThreshold
      }).getOrElse(maxK - 1) + 1
    val pcaRes = new PCA()
      .setInputCol(inColName)
      .setOutputCol(outColName)
      .setK(significantPCADimensions)
      .fit(series)

    (new TSS(pcaRes.transform(series)), pcaRes.pc)
  }

  def getPeriodograms(tss: TSS)(implicit ss: SparkSession) = {

    val withFourierTable: TSS =
    tss.addZNormalized("zseries", TSS.SERIES_COLNAME, 0.0001)
    .addDFT("dft", "zseries")
    .addDFTFrequencies("dftFreq", TSS.SERIES_COLNAME, TSS.TIMEGRANULARITY_COLNAME)
    .addDFTPeriodogram("dftPeriodogram", "dft")

     val meanFourierFrequencyStep = withFourierTable
      .colSeqFirstStep("dftFreq")
      .agg(functions.mean("value"))
      .first.getDouble(0)

    val newInterpolationSamplePoints = (0 until 30).map(_.toDouble * meanFourierFrequencyStep)
    val minMaxAndMaxMinFourierFrequency = withFourierTable.series.select(min(array_max(col("dftFreq"))), max(array_min(col("dftFreq")))).first
    val minMaxFourierFrequency = minMaxAndMaxMinFourierFrequency.getDouble(0)
    val maxMinFourierFrequency = minMaxAndMaxMinFourierFrequency.getDouble(1)
    val keptInterpolationSamplePoints: Array[Double] = newInterpolationSamplePoints.filter(x => x < minMaxFourierFrequency && x > maxMinFourierFrequency).toArray

    val interpolatedFourierTSS = withFourierTable
    .addConstant("interpolatedDFTFreq", keptInterpolationSamplePoints)
    .addLinearInterpolationPoints("interpolatedDFTPeriodogram", "dftFreq", "dftPeriodogram", keptInterpolationSamplePoints)

    val logScaledTSS = interpolatedFourierTSS.addUDFColumn("logInterpolatedDFTPeriodogram",
      "interpolatedDFTPeriodogram",
      functions.udf(tssFunctions.log10(1D)
        .andThen((seq: Seq[Double]) => {Vectors.dense(seq.toArray)})))
      .repartition(200)
    val scaledTSS: TSS = logScaledTSS.addColScaled("logInterpolatedDFTPeriodogram_ScaledVecColScaled",
        "logInterpolatedDFTPeriodogram",true,true)
    val seqScaledTSS = scaledTSS.addSeqFromMLVector("periodogram",
      "logInterpolatedDFTPeriodogram_ScaledVecColScaled")
    val series = seqScaledTSS.series

    val outputDf = series.select(
      seqScaledTSS.getDecoratorColumn("scenario_id").alias("scenario_id"),
      seqScaledTSS.getDecoratorColumn("varName").alias("varName"),
      col("periodogram")).rdd

    outputDf.map(row => (
      row.getString(0).toInt,
      row.getString(1).toInt,
      row.getSeq[Double](2).toArray.toList)
    )
  }

  def getPcaAndLoadings(dataRDD: RDD[(Int, Int, List[Double])])(implicit ss: SparkSession):
  (RDD[(Int, Int, BzDenseVector[Double])], BzDenseMatrix[Double]) = {

    val dfWithSchema = ss.createDataFrame(dataRDD).toDF("scenario_id", "varName", "periodogram")

    val tss = new TSS(dfWithSchema, forceIds = false)

    val tssVec = tss.addMLVectorized("periodogramMLVec", "periodogram")

    val (joinedTSS, loadings): (TSS, org.apache.spark.ml.linalg.DenseMatrix) =
      addPCA(tssVec.series,"logInterpolatedDFTPeriodogram_PCAVec",
        "periodogramMLVec",
        20,0.99, None,None)

    //Scale PCA results to enforce the correct relative importance of each feature to the afterwards weighting
    val scaledJoinedTSS = joinedTSS.addColScaled("logInterpolatedDFTPeriodogram_ColScaledPCAVec",
      "logInterpolatedDFTPeriodogram_PCAVec", true,true)
      .addSeqFromMLVector("pcaCoordinatesV", "logInterpolatedDFTPeriodogram_ColScaledPCAVec")
      //Drop intermediate columns for cleaner output
      .drop("logInterpolatedDFTPeriodogram_ColScaledPCAVec", "logInterpolatedDFTPeriodogram_PCAVec", "periodogram")

    val series = scaledJoinedTSS.select("scenario_id","varName","pcaCoordinatesV").series
    val pcaCoefs = series.select(col("scenario_id"),
          col("varName"),
          col("pcaCoordinatesV")).rdd

    val pcaCoefsRDDs = pcaCoefs.map(row => (
      row.getInt(0),
      row.getInt(1),
      BzDenseVector[Double](row.getSeq[Double](2).toArray))
    )

    (pcaCoefsRDDs, matrixToDenseMatrix(loadings).t)

  }

}

