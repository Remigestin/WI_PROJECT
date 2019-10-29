import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils

object FirstSparkApplication extends App {

  //Only for windows user I think
  System.setProperty("hadoop.home.dir", "C:\\winutils")

  val spark = SparkSession.builder
    .master("local[*]")
    .appName("FirstSparkApplication")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")


  val jsonData = spark.read.json("data-students.json")

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //DATA CLEANING :

  //how to count all the line for a dataFrame
  //println("number total of row =  " + jsonData.count())
 // println("number of bidFloor null = " + jsonData.filter("bidFloor is NULL").count())
  //jsonData.groupBy("label").count.show()
  //try to supp null value for bidFloor
  val cleanJson = jsonData.na.drop("any", Seq("bidFloor"))
  //println("hey le nb null part 2 " + cleanJson.filter("bidFloor is NULL").count())
  //cleanJson.groupBy("label").count.show()
  //println("number total of row =  " + cleanJson.count())

  //appOrSite, media, label les passer sous indexer

  //Select only the good columns
  val finalJson = cleanJson.select("bidFloor", "media", "label")


  //indexJson.show(100000)
  //finalJson.printSchema()
  //println("-----------------")
  //indexerMedia.printSchema()
  //cleanData.printSchema()

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Tree classifier


  val df2 = finalJson.selectExpr("cast(label AS int) label",
    "bidFloor",
    "media")

  //Parse the data
  val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3))


  //We define a StringIndexers for the categorical variable media
  val indexerMedia = new StringIndexer()
    .setInputCol("media")
    .setOutputCol("mediaIndex")

  //We define the assembler to collect the columns into a new column with a single vector - "features"
  val assembler = new VectorAssembler()
    .setInputCols(Array("bidFloor", "mediaIndex"))
    .setOutputCol("features")

  //For the regression we'll use the Gradient-boosted tree estimator
  val gbt = new GBTRegressor()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setPredictionCol("Predicted " + "label")
    .setMaxIter(50)

  //Construct the pipeline
  val pipeline = new Pipeline().setStages(Array(indexerMedia, assembler, gbt))
  //We fit our DataFrame into the pipeline to generate a model
  val model = pipeline.fit(trainingData)
  //We'll make predictions using the model and the test data
  val predictions = model.transform(testData)

  //This will evaluate the error/deviation of the regression using the Root Mean Squared deviation
  val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("Predicted " + " label")
    .setMetricName("rmse")
  //We compute the error using the evaluator
  val error = evaluator.evaluate(predictions)
  println(error)

  spark.stop

}
