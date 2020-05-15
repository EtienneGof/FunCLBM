
package FunCLBMSpark

import breeze.linalg.{*, DenseMatrix, DenseVector, argmax}
import breeze.numerics.abs
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.stat.distribution.MultivariateGaussian
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import Tools._
import OutputResults._
import scala.collection.mutable.ListBuffer
import scala.util.{Failure, Random, Success, Try}


class CondLatentBlock private(private var KVec: List[Int],
                              private var maxIterations: Int,
                              private var maxBurninIterations: Int,
                              private var fullCovarianceHypothesis: Boolean = true,
                              private var updateLoadings: Boolean = false,
                              private var updateLoadingStrategy: Double = 0,
                              private var seed: Long) extends Serializable {

  val precision = 1e-5

  /**
    * Constructs a default instance. The default parameters are {k: 2, convergenceTol: 0.01,
    * maxIterations: 100, seed: random}.
    */
  def this() = this(List(2,2), 6, 6, seed = Random.nextLong())

  // an initializing model can be provided rather than using the
  // default random starting point
  private var providedInitialModel: Option[CondLatentBlockModel] = None
  private var providedInitialColPartition: Option[List[Int]] = None

  /**
    * Set the initial GMM starting point, bypassing the random initialization.
    * You must call setK() prior to calling this method, and the condition
    * (model.k == this.k) must be met; failure will result in an IllegalArgumentException
    */
  def setInitialModel(model: CondLatentBlockModel): this.type = {
    require(model.KVec == KVec,
      s"Mismatched row cluster number (model.KVec ${model.KVec} != KVec $KVec)")
    providedInitialModel = Some(model)
    this
  }

  def setInitialColPartition(colPartition: List[Int]): this.type = {
    val uniqueCol = colPartition.distinct
    require(uniqueCol.length == KVec.length,
      s"Mismatched column cluster number (colPartition.distinct.length ${uniqueCol.length} != KVec.length ${KVec.length})")
    providedInitialColPartition = Some(colPartition)
    this
  }


  /**
    * Return the user supplied initial GMM, if supplied
    */
  def getInitialModel: Option[CondLatentBlockModel] = providedInitialModel

  /**
    * Set the number of Gaussians in the mixture model.  Default: 2
    */
  def setKVec(KVec: List[Int]): this.type = {
    require(KVec.forall(_ > 0),
      s"Every numbers of row clusters must be positive but got $KVec")
    this.KVec = KVec
    this
  }

  /**
    * Return the number of row cluster number in the latent block model
    */
  def getKVec: List[Int] = KVec


  /**
    * Return the number of column cluster number in the latent block model
    */
  def getL: Int = KVec.length

  /**
    * Set Whether the loadings are updated at each M step, or not
    */
  private def setUpdateLoadings(updateLoadings: Boolean): this.type = {
    this.updateLoadings = updateLoadings
    this
  }
  private def getUpdateLoadings: Boolean = updateLoadings

  def setUpdateLoadingStrategy[T](updateLoadingStrategy: T): this.type = {
    val updateLoadingStrategyTmp = updateLoadingStrategy.toString
    require(List("always","never").contains(updateLoadingStrategyTmp) ||
      (Tools.isInteger(updateLoadingStrategyTmp) &
        updateLoadingStrategyTmp.toInt >= 0), "updateLoadingStrategy should be an int >= 0 or ('always', 'never'))")

    this.updateLoadingStrategy = updateLoadingStrategyTmp match {
      case "always" => 0D
      case "never" => Double.PositiveInfinity
      case _ => updateLoadingStrategyTmp.toInt.toDouble
    }
    this
  }
  def getUpdateLoadingStrategy: Double = updateLoadingStrategy


  /**
    * Set the maximum number of iterations allowed. Default: 100
    */
  def setMaxIterations(maxIterations: Int): this.type = {
    require(maxIterations > 0,
      s"Maximum of iterations must be strictly positive but got $maxIterations")
    this.maxIterations = maxIterations
    this
  }

  /**
    * Set the maximum number of iterations allowed. Default: 100
    */
  def setMaxBurninIterations(maxBurninIterations: Int): this.type = {
    require(maxBurninIterations >= 0,
      s"Maximum of Burn-in iterations must be positive or zero but got $maxBurninIterations")
    this.maxBurninIterations = maxBurninIterations
    this
  }

  /**
    * Return the maximum number of iterations allowed
    */
  def getMaxIterations: Int = maxIterations

  /**
    * Return the maximum number of iterations allowed
    */
  def getMaxBurninIterations: Int = maxBurninIterations

  /**
    * Set the number of Gaussians in the mixture model.  Default: 2
    */
  def setFullCovarianceHypothesis(fullCovarianceHypothesis: Boolean): this.type = {
    this.fullCovarianceHypothesis = fullCovarianceHypothesis
    this
  }

  /**
    * Return the number of row cluster number in the latent block model
    */
  def getFullCovarianceHypothesis: Boolean = fullCovarianceHypothesis

  /**
    * Set the random seed
    */
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  /**
    * Return the random seed
    */
  def getSeed: Long = seed

  def computeMeanModels(models: List[CondLatentBlockModel]): CondLatentBlockModel = {

    val nTotalIter = models.length.toDouble
    val meanProportionRows: List[List[Double]] = models.head.proportionsRows.indices.map(idx => {
      models.map(model =>
        DenseVector(model.proportionsRows(idx).toArray) / nTotalIter).reduce(_ + _).toArray.toList
    }).toList

    val meanProportionCols: List[Double] = models.map(model =>
      DenseVector(model.proportionsCols.toArray) / nTotalIter).reduce(_ + _).toArray.toList

    val meanLoadings: List[List[DenseMatrix[Double]]] = KVec.indices.map(l => {
      (0 until KVec(l)).map(k_l => {
        models.map(m => m.loadings(l)(k_l)).reduce(_ + _) / nTotalIter
      }).toList
    }).toList

    val meanMu: List[List[DenseVector[Double]]] =
      KVec.indices.map(l => {
        (0 until KVec(l)).map(k_l => {
          DenseVector(models.map(model => DenseVector(model.gaussians(l)(k_l).mean.toArray)).reduce(_ + _).toArray) / nTotalIter
        }).toList
      }).toList

    val meanGaussians: List[List[MultivariateGaussian]] = KVec.indices.map(l => {
      (0 until KVec(l)).map(k_l => {
        val covMat = models.map(m => matrixToDenseMatrix(m.gaussians(l)(k_l).cov)).reduce(_ + _) / nTotalIter
        new MultivariateGaussian(Vectors.dense(meanMu(l)(k_l).toArray), denseMatrixToMatrix(covMat))
      }).toList
    }).toList

    CondLatentBlockModel(meanProportionRows, meanProportionCols, meanLoadings, meanGaussians)
  }

  def initAndRunTry(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                    n: Int,
                    p: Int,
                    EMMethod:String = "SEMGibbs",
                    nTry: Int = 1,
                    nTryMax: Int = 50,
                    initMethod: String = "randomPartition",
                    verbose: Boolean=false)(implicit ss: SparkSession): Map[String,Product] = {


    var t0 = System.nanoTime()
    var logLikelihoodList: ListBuffer[Double] = new ListBuffer[Double]() += Double.NegativeInfinity
    var ICL: ListBuffer[Double] = new ListBuffer[Double]() += Double.NegativeInfinity

    if(nTry > nTryMax){
      return Map("Model" -> new CondLatentBlockModel(),
        "RowPartition" -> List(List.fill(n)(0)),
        "ColPartition" -> List.fill(p)(0),
        "LogLikelihood" -> logLikelihoodList.toList,
        "ICL" -> ICL.toList)
    }

    Try(this.initAndLaunch(periodogram, n,p,EMMethod, initMethod = initMethod, verbose=verbose)) match {
      case Success(v) =>
        if(verbose){println()
          Tools.printTime(t0, EMMethod+" FunCLBM Spark")}
        Success(v).get.asInstanceOf[Map[String, Product with Serializable]]
      case Failure(e) =>
        if(verbose){
          if(nTry==1){
            print("Algorithm "+ EMMethod+" didn't converge to an appropriate solution, trying again..\n" +
              "n° try: "+nTry.toString+"")
          } else {print(", "+nTry.toString)}}
        this.initAndRunTry(periodogram, n, p,
          EMMethod,
          nTry+1,
          nTryMax,
          initMethod=initMethod,
          verbose=verbose)
    }
  }

  def initAndLaunch(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                    n:Int,
                    p:Int,
                    EMMethod: String,
                    verbose:Boolean=true,
                    initMethod: String = "randomPartition")(implicit ss: SparkSession): Map[String,Product]= {

    val (initialModel, initialColPartition) = providedInitialModel match {
      case Some(model) => {
        require(initMethod.isEmpty,
          s"An initial model has been provided but initMethod argument has also been set to $initMethod. " +
            s"Please make a choice: do not set an initMethod or do not provide an initial model.")
        (model,
          providedInitialColPartition match {
            case Some(colPartition) => colPartition
            case None => (0 until p).map(j => sample(model.proportionsCols)).toList
          })
      }
      case None => {
        Initialization.initialize(periodogram,this,EMMethod,n,p,verbose,initMethod)
      }
    }

    launchEM(periodogram, EMMethod, initialColPartition,initialModel,n, p,verbose)
  }

  def launchEM(periodogram: RDD[(Int, Array[DenseVector[Double]])],
               EMMethod: String,
               initialColPartition: List[Int],
               initialModel: CondLatentBlockModel,
               n:Int,
               p:Int,
               verbose:Boolean=false)(implicit ss: SparkSession): Map[String,Product] = {

    require(List("SEMGibbs").contains(EMMethod),
      "EM Method provided "+EMMethod+" is not available. Possible options: SEMGibbs")
    EMMethod match {
      case "SEMGibbs" => SEMGibbs(periodogram, initialColPartition, initialModel,n,p, verbose)
    }
  }

  def run(periodogram: RDD[(Int, Array[DenseVector[Double]])],
          EMMethod: String= "SEMGibbs",
          verbose: Boolean = false,
          nConcurrent:Int=1,
          nTryMaxPerConcurrent:Int=20,
          initMethod: String = "randomPartition")(implicit ss: SparkSession): Map[String,Product] = {
    val n:Int = periodogram.count().toInt
    val p:Int = periodogram.take(1).head._2.length

    val allRes = (0 until nConcurrent).map(nTry => {
      if(verbose){println("> n° launch "+(1+nTry).toString+"/"+nConcurrent.toString)}
      this.initAndRunTry(periodogram, n, p, EMMethod, nTryMax = nTryMaxPerConcurrent, initMethod = initMethod, verbose=verbose)
    }).toList
    val allLikelihoods: DenseVector[Double] = DenseVector(allRes.map(_("LogLikelihood").asInstanceOf[List[Double]].last).toArray)

    allRes(argmax(allLikelihoods))
  }

  def SEMGibbs(periodogram: RDD[(Int, Array[DenseVector[Double]])],
               initialColPartition: List[Int],
               initialModel: CondLatentBlockModel,
               n:Int,
               p:Int,
               verbose:Boolean=true,
               withCachedModels: Boolean= true)(implicit ss: SparkSession): Map[String,Product] = {

    require(this.fullCovarianceHypothesis, "in SEM-Gibbs mode, indep. covariance hyp. is not yet available," +
      " please set latentBlock.fullCovarianceHypothesis to true")
    val iterBeginUpdateLoadings = this.updateLoadingStrategy
    this.setUpdateLoadings(false)
    var precPeriodogram: RDD[(Int, Array[DenseVector[Double]], List[Int])] = periodogram.map(r => (r._1, r._2, List(0,0)))
    var precColPartition = initialColPartition
    precPeriodogram = initialModel.drawRowPartition(precPeriodogram, precColPartition)
    var precModel = initialModel

    var completeLogLikelihoodList: ListBuffer[Double] = new ListBuffer[Double]():+ Double.NegativeInfinity :+ precModel.completelogLikelihood(precPeriodogram, precColPartition)
    var cachedModels = List[CondLatentBlockModel]():+precModel

    //        val newRowPartitionPerRow = precPeriodogram.map(_._3).collect()
    //        val newRowPartition:List[List[Int]] = KVec.indices.map(l =>
    //          newRowPartitionPerRow.indices.map(i => newRowPartitionPerRow(i)(l)).toList).toList
    if(verbose){println(">>>>> Initial model")}
    //        newRowPartition.foreach(println)
    //        println(precColPartition)
    //        println(precModel.gaussians.map(_.length))
    //        precModel.gaussians.flatten.foreach(m => println(m.mean))
    var t0 = System.nanoTime
    var iter =0

    do {
      iter +=1
      if(iter>iterBeginUpdateLoadings){this.setUpdateLoadings(true)}
      if(verbose){println(">>>>> SEM Gibbs iteration: "+iter.toString)}
      val (newData, newColPartition) = precModel.StochasticExpectationStep(
        precPeriodogram,
        precColPartition,
        p,
        verbose = verbose)
      if(verbose){t0 = printTime(t0, "SE")}

      //
      //      val newRowPartitionPerRow = newData.map(_._3).collect()
      //      val newRowPartition:List[List[Int]] = KVec.indices.map(l =>
      //        newRowPartitionPerRow.indices.map(i => newRowPartitionPerRow(i)(l)).toList).toList
      //      newRowPartition.foreach(println)
      //      println(newColPartition)

      val newModel = precModel.SEMGibbsMaximizationStep(newData,
        newColPartition, n, verbose, updateLoadings = this.updateLoadings)
      //            println(precModel.gaussians.map(_.length))
      //            precModel.gaussians.flatten.foreach(m => println(m.mean))
      if(verbose){t0 = printTime(t0, "Maximization")}

      precModel = newModel
      precPeriodogram = newData
      precColPartition = newColPartition
      completeLogLikelihoodList += precModel.completelogLikelihood(precPeriodogram, precColPartition)

      cachedModels = cachedModels :+ precModel
      if(verbose){println("Loglikelihood: "+completeLogLikelihoodList.last.toString)}

    } while (iter < (maxBurninIterations+maxIterations) &
      !(abs(completeLogLikelihoodList.last - completeLogLikelihoodList.dropRight(1).last)<precision &
        abs(completeLogLikelihoodList.last - completeLogLikelihoodList.dropRight(2).last)<precision)  )

    val iclList = completeLogLikelihoodList.map(ll => precModel.ICL(ll,n,p,true))
    val rowMembershipPerRow: List[List[Int]] = precPeriodogram.map(row => (row._1, row._3)).collect().sortBy(_._1).map(_._2).toList
    val rowPartition: List[List[Int]] = precModel.KVec.indices.map(l =>
      rowMembershipPerRow.map(rowMembership => rowMembership(l))).toList

    var res = if (!(abs(completeLogLikelihoodList.last - completeLogLikelihoodList.dropRight(1).last)<precision &
      abs(completeLogLikelihoodList.last - completeLogLikelihoodList.dropRight(2).last)<precision)) {
      Map("Model" -> computeMeanModels(cachedModels.drop(1+maxBurninIterations)),
        "RowPartition" -> rowPartition,
        "ColPartition" -> precColPartition,
        "LogLikelihood" -> completeLogLikelihoodList.toList.drop(1),
        "ICL" -> iclList.toList.drop(1))
    } else {
      Map("Model" -> cachedModels.last,
        "RowPartition" -> rowPartition,
        "ColPartition" -> precColPartition,
        "LogLikelihood" -> completeLogLikelihoodList.toList.drop(1),
        "ICL" -> iclList.toList.drop(1))
    }

    if(withCachedModels) {res += ("CachedModels" -> cachedModels)}

    res
  }

}
