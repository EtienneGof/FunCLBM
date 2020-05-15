
package FunLBMSpark

import FunCLBMSpark.Tools
import FunCLBMSpark.Tools._
import breeze.linalg.{DenseMatrix, DenseVector, argmax}
import breeze.numerics.abs
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.stat.distribution.MultivariateGaussian
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ListBuffer
import scala.util.{Failure, Random, Success, Try}


class LatentBlock private(private var K: Int,
                          private var L: Int,
                          private var maxIterations: Int,
                          private var maxBurninIterations: Int,
                          private var updateLoadings: Boolean = false,
                          private var updateLoadingStrategy: Double = 0,
                          private var fullCovarianceHypothesis: Boolean = true,
                          private var seed: Long) extends Serializable {

  val precision = 1e-5

  /**
    * Constructs a default instance. The default parameters are {k: 2, convergenceTol: 0.01,
    * maxIterations: 100, seed: random}.
    */
  def this() = this(2, 2 , 20, 20, seed = Random.nextLong())

  // an initializing model can be provided rather than using the
  // default random starting point
  private var providedInitialModel: Option[LatentBlockModel] = None

  /**
    * Set the initial GMM starting point, bypassing the random initialization.
    * You must call setK() prior to calling this method, and the condition
    * (model.k == this.k) must be met; failure will result in an IllegalArgumentException
    */
  def setInitialModel(model: LatentBlockModel): this.type = {
    require(model.K == K,
      s"Mismatched row cluster number (model.KVec ${model.K} != KVec $K)")
    providedInitialModel = Some(model)
    this
  }

  /**
    * Return the user supplied initial GMM, if supplied
    */
  def getInitialModel: Option[LatentBlockModel] = providedInitialModel

  /**
    * Set the number of Gaussians in the mixture model.  Default: 2
    */
  def setK(K: Int): this.type = {
    require(K>0,
      s"Row cluster number must be positive but got $K")
    this.K = K
    this
  }

  /**
    * Set the number of Gaussians in the mixture model.  Default: 2
    */
  def setL(L: Int): this.type = {
    require(L>0,
      s"Row cluster number must be positive but got $L")
    this.L = L
    this
  }

  /**
    * Return the number of row cluster number in the latent block model
    */
  def getK: Int = K


  /**
    * Return the number of column cluster number in the latent block model
    */
  def getL: Int = L

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
      (FunCLBMSpark.Tools.isInteger(updateLoadingStrategyTmp) &
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

  def computeMeanModels(models: List[LatentBlockModel]): LatentBlockModel = {

    val meanProportionRows: List[Double] = (models.map(model =>
      DenseVector(model.proportionsRows.toArray)).reduce(_+_)/models.length.toDouble).toArray.toList

    val meanProportionCols: List[Double] = (models.map(model =>
      DenseVector(model.proportionsCols.toArray)).reduce(_+_)/models.length.toDouble).toArray.toList

    val meanMu: DenseMatrix[DenseVector[Double]] = DenseMatrix.tabulate(K,L){ (k, l)=>{
      models.map(m => DenseVector(m.gaussians(k)(l).mean.toArray)).reduce(_+_)/models.length.toDouble
    }}

    val meanCovariance: DenseMatrix[DenseMatrix[Double]] = DenseMatrix.tabulate(K,L){ (k, l)=>{
      models.map(m => DenseMatrix(m.gaussians(k)(l).cov.toArray).reshape(meanMu(0,0).length,meanMu(0,0).length)).reduce(_+_)/models.length.toDouble
    }}

    val meanGaussians: List[List[MultivariateGaussian]] = (0 until K).map(k => {
      (0 until L).map(l => {
        new MultivariateGaussian(
          Vectors.dense(meanMu(k,l).toArray),
          denseMatrixToMatrix(meanCovariance(k,l)
          ))
      }).toList
    }).toList

    new LatentBlockModel(meanProportionRows, meanProportionCols, models.head.loadings, meanGaussians)
  }

  def initAndRunTry(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                    n: Int,
                    p: Int,
                    EMMethod:String = "SEMGibbs",
                    nTry: Int = 1,
                    nTryMax: Int = 50,
                    initMethod: String = "",
                    verbose: Boolean=false)(implicit ss: SparkSession): Map[String,Product] = {

    var t0 = System.nanoTime()
    var logLikelihoodList: ListBuffer[Double] = new ListBuffer[Double]() += Double.NegativeInfinity
    var ICL: ListBuffer[Double] = new ListBuffer[Double]() += Double.NegativeInfinity


    if(nTry > nTryMax){
      return Map("Model" -> new LatentBlockModel(),
        "ColPartition" -> List.fill(p)(0),
        "LogLikelihood" -> logLikelihoodList.toList,
        "ICL" -> ICL.toList)
    }

    Try(this.initAndLaunch(periodogram, n,p,EMMethod, initMethod = initMethod, verbose=verbose)) match {
      case Success(v) =>
        if(verbose){println()
          printTime(t0, EMMethod+" FunLBM Spark")}
        Success(v).get.asInstanceOf[Map[String, Product with Serializable]]
      case Failure(e) =>
        if(verbose){
          if(nTry==1){
            print("Algorithm "+ EMMethod+" didn't converge to an appropriate solution, trying again..\n" +
              "n° try: "+nTry.toString+"")
          } else {print(", "+nTry.toString)}}
        this.initAndRunTry(periodogram, n, p, EMMethod, nTry+1,nTryMax, initMethod=initMethod, verbose=verbose)
    }
  }

  def initAndLaunch(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                    n:Int,
                    p:Int,
                    EMMethod: String,
                    verbose:Boolean=true,
                    initMethod: String = "")(implicit ss: SparkSession): Map[String,Product]= {

    val (initialModel, initialColPartition) = providedInitialModel match {
      case Some(model) => {
        require(initMethod.isEmpty,
          s"An initial model has been provided but initMethod argument has also been set to $initMethod. " +
            s"Please make a choice: do not set an initMethod or do not provide an initial model.")
        (model,
          (0 until p).map(j => sample(model.proportionsCols)).toList)
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
               initialModel: LatentBlockModel,
               n:Int,
               p:Int,
               verbose:Boolean=false)(implicit ss: SparkSession): Map[String,Product] = {

    require(List("SEMGibbs","VariationalEM").contains(EMMethod),
      "EM Method provided "+EMMethod+" is not available. Possible options: SEMGibbs")
    EMMethod match {
      case "SEMGibbs" => SEMGibbs(periodogram, initialColPartition, initialModel,n,p, verbose)
    }
  }

  def run(periodogram: RDD[(Int, Array[DenseVector[Double]])],
          EMMethod: String= "SEMGibbs",
          verbose: Boolean = false,
          nConcurrent:Int=3,
          nTryMaxPerConcurrent:Int=20,
          initMethod: String = "random")(implicit ss: SparkSession): Map[String,Product] = {

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
               initialModel: LatentBlockModel,
               n:Int,
               p:Int,
               verbose:Boolean=true,
               withCachedModels: Boolean= true)(implicit ss: SparkSession): Map[String,Product] = {

    require(this.fullCovarianceHypothesis, "in SEM-Gibbs mode, indep. covariance hyp. is not yet available," +
      " please set latentBlock.fullCovarianceHypothesis to true")
    val iterBeginUpdateLoadings = this.updateLoadingStrategy
    this.setUpdateLoadings(false)

    var precPeriodogram: RDD[(Int, Array[DenseVector[Double]], Int)] = periodogram.map(r => (r._1, r._2, 0))
    var precColPartition = initialColPartition
    precPeriodogram = initialModel.drawRowPartition(precPeriodogram, precColPartition)
    var precModel = initialModel

    var completeLogLikelihoodList: ListBuffer[Double] = new ListBuffer[Double]():+ Double.NegativeInfinity :+
      precModel.completelogLikelihood(precPeriodogram, precColPartition)
    var cachedModels = List[LatentBlockModel]():+precModel
    val newRowPartition = precPeriodogram.map(_._3).collect().toList

    var iter =0
    do {
      iter +=1
      if(iter>iterBeginUpdateLoadings){this.setUpdateLoadings(true)}
//      if(verbose){println(">>> iter: "+iter.toString)}
      val (newData, newColPartition) = precModel.StochasticExpectationStep(
        precPeriodogram,
        precColPartition,
        p,
        verbose = verbose)

      val newRowPartition = precPeriodogram.map(_._3).collect().toList
      val newModel = precModel.SEMGibbsMaximizationStep(newData, newColPartition, n, verbose)
      precModel = newModel
      precPeriodogram = newData
      precColPartition = newColPartition
      cachedModels = cachedModels :+ precModel
      completeLogLikelihoodList += precModel.completelogLikelihood(precPeriodogram, precColPartition)

    } while (iter < (maxBurninIterations+maxIterations) &
      !(abs(completeLogLikelihoodList.last - completeLogLikelihoodList.dropRight(1).last)<precision &
        abs(completeLogLikelihoodList.last - completeLogLikelihoodList.dropRight(2).last)<precision)  )

    val iclList = completeLogLikelihoodList.map(ll => precModel.ICL(ll,n,p,true))
    val rowPartition: List[Int] = precPeriodogram.map(row => (row._1, row._3)).collect().sortBy(_._1).map(_._2).toList

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
