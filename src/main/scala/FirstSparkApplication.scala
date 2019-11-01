import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.{GBTRegressor, LinearRegression,LinearRegressionModel}
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
  // Index labels, adding metadata to the label column.
  // Fit on whole dataset to include all labels in index.

  val df2 = finalJson.selectExpr("cast(label AS STRING) label",
    "bidFloor",
    "media")

  // Split the data into training and test sets (30% held out for testing)
  val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3))

  val labelIndexer = new StringIndexer()
    .setInputCol("label")
    .setOutputCol("indexedLabel")
    .fit(df2)


  //We define a StringIndexers for the categorical variable media
  val indexerMedia = new StringIndexer()
    .setInputCol("media")
    .setOutputCol("mediaIndex")
    .fit(df2)


  val assembler = new VectorAssembler()
    .setInputCols(Array("bidFloor", "mediaIndex"))
    .setOutputCol("features")


  // Train a DecisionTree model.
  val lr = new LinearRegression()
    .setLabelCol("label")
    .setFeaturesCol("features")



  // Convert indexed labels back to original labels.
 /* val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(labelIndexer.labels)
*/

  // Chain indexers and tree in a Pipeline
  val pipeline = new Pipeline().setStages(Array(labelIndexer,indexerMedia, assembler, lr))
  //val pipeline = new Pipeline().setStages(Array(indexerMedia, assembler, lr))
  // Train model.  This also runs the indexers.
  val model = pipeline.fit(trainingData)


  //Show model results
  val linRegModel = model.stages(1).asInstanceOf[LinearRegressionModel]
  linRegModel.summary.residuals.show()

  // Make predictions.
  /*val predictions = model.transform(testData)

  // Select example rows to display.
  predictions.select("predictedLabel", "label", "features").show(5)

  // Select (prediction, true label) and compute test error
  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("precision")
  val accuracy = evaluator.evaluate(predictions)
  println("Test Error = " + (1.0 - accuracy))
*/
  spark.stop

}
