package FunCLBMSpark

import java.io.{BufferedWriter, FileWriter}

import breeze.linalg.{*, DenseMatrix}
import com.opencsv.CSVWriter
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

import scala.collection.JavaConverters._
import scala.util.{Failure, Try}

object OutputResults {

  def addPrefix(lls: List[List[String]]): List[List[String]] =
    lls.foldLeft((1, List.empty[List[String]])){
      case ((serial: Int, acc: List[List[String]]), value: List[String]) =>
        (serial + 1, (serial.toString +: value) +: acc)
    }._2.reverse


  def writeMatrixStringToCsv(fileName: String, Matrix: DenseMatrix[String], append: Boolean = false): Unit = {
    val rows : List[List[String]] = Matrix(*, ::).map(dv => dv.toArray.map(_.toString).toList).toArray.toList
    writeCsvFile(fileName, addPrefix(rows), append=append)
  }

  def writeMatrixDoubleToCsv(fileName: String, Matrix: DenseMatrix[Double], withHeader:Boolean=true): Unit = {
    val header: List[String] = List("id") ++ (0 until Matrix.cols).map(_.toString).toList
    val rows : List[List[String]] = Matrix(*, ::).map(dv => dv.toArray.map(_.toString).toList).toArray.toList
    if(withHeader){
      writeCsvFile(fileName, addPrefix(rows), header)
    } else {
      writeCsvFile(fileName, addPrefix(rows))
    }
  }

  def writeMatrixIntToCsv(fileName: String, Matrix: DenseMatrix[Int], withHeader:Boolean=true): Unit = {
    val header: List[String] = List("id") ++ (0 until Matrix.cols).map(_.toString).toList
    val rows : List[List[String]] = Matrix(*, ::).map(dv => dv.toArray.map(_.toString).toList).toArray.toList
    if(withHeader){
      writeCsvFile(fileName, addPrefix(rows), header)
    } else {
      writeCsvFile(fileName, addPrefix(rows))
    }
  }

  def writeCsvFile(fileName: String,
                   rows: List[List[String]],
                   header: List[String] = List.empty[String],
                   append:Boolean=false
                  ): Try[Unit] =
  {
    val content = if(header.isEmpty){rows} else {header +: rows}
    Try(new CSVWriter(new BufferedWriter(new FileWriter(fileName, append)))).flatMap((csvWriter: CSVWriter) =>
      Try{
        csvWriter.writeAll(
          content.map(_.toArray).asJava
        )
        csvWriter.close()
      } match {
        case f @ Failure(_) =>
          // Always return the original failure.  In production code we might
          // define a new exception which wraps both exceptions in the case
          // they both fail, but that is omitted here.
          Try(csvWriter.close()).recoverWith{
            case _ => f
          }
        case success =>
          success
      }
    )
  }

  def writeModel(SEMGibbsOutput: Map[String, Product], pathOutput: String)={

    val outputModel = SEMGibbsOutput("Model").asInstanceOf[CondLatentBlockModel]
    println(outputModel.proportionsCols)
    println(outputModel.proportionsRows)

    val outputContent = outputModel.proportionsCols.indices.map(j => {
      outputModel.proportionsRows(j).indices.map(i => {
        List(i.toString,
          j.toString,
          outputModel.proportionsRows(j)(i).toString,
          outputModel.proportionsCols(j).toString,
          outputModel.gaussians(j)(i).mean.toArray.mkString(":"),
          outputModel.gaussians(j)(i).cov.toArray.mkString(":"))
      }).toList
    }).reduce(_++_)

    val header: List[String] = List("id","rowCluster","colCluster","rowProportion", "colProportion", "mean","cov")
    OutputResults.writeCsvFile(pathOutput,OutputResults.addPrefix(outputContent),header)
  }

  def writeData(dataRDD: RDD[Row],
                SEMGibbsOutput: Map[String, Product],
                pathOutput: String,
                mapIterationId: scala.collection.Map[String, Int],
                mapVarName: scala.collection.Map[String, Int])={

    val rowPartition = SEMGibbsOutput("RowPartition").asInstanceOf[List[List[Int]]]
    val colPartition = SEMGibbsOutput("ColPartition").asInstanceOf[List[Int]]

    val reverseMapIterationId = for ((k,v) <- mapIterationId) yield (v, k)
    val reverseMapVarName = for ((k,v) <- mapVarName) yield (v, k)

    val outputContent = dataRDD.map(row => {
      val iterationId = row.getString(1)
      val varNameId = row.getString(0)
      val l = colPartition(varNameId.toInt)
      val k_l = rowPartition(l)(iterationId.toInt)
      List(
        reverseMapIterationId(iterationId.toInt),
        reverseMapVarName(varNameId.toInt),
        k_l.toString,
        l.toString,
        row(2).asInstanceOf[Seq[Double]].toArray.mkString(":"),
        row(3).asInstanceOf[Seq[Double]].toArray.mkString(":")
      )}).collect().toList

    OutputResults.writeCsvFile(pathOutput,outputContent)
  }

}
