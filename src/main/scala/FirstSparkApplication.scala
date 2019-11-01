import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession

object FirstSparkApplication extends App {

  //Only for windows user I think
  System.setProperty("hadoop.home.dir", "C:\\winutils")

  val spark = SparkSession.builder
    .master("local[*]")
    .appName("FirstSparkApplication")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  // Read json data
  val jsonData = spark.read.json("data-students.json")


  // DATA CLEANING
  val cleanedData = Cleaner.generalClean(jsonData)

  //appOrSite, media, label les passer sous indexer

  // Select only the needed columns
  val finalData = cleanedData.select(
    "label",
    "bidFloor", "media", "appOrSite",
    "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10",
    "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19",
    "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26")

  // Index labels, adding metadata to the label column.
  val labelIndexer = new StringIndexer()
    .setInputCol("label")
    .setOutputCol("indexedLabel")
    .fit(finalData)

  //We define a StringIndexers for the categorical variable media
  val indexerMedia = new StringIndexer()
    .setInputCol("media")
    .setOutputCol("mediaIndex")

  //We define a StringIndexers for the categorical variable media
  val indexerAppOrSite = new StringIndexer()
    .setInputCol("appOrSite")
    .setOutputCol("appOrSiteIndex")

  // Create features column
  val assembler = new VectorAssembler()
    .setInputCols(Array("bidFloor", "mediaIndex", "appOrSiteIndex",
      "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10",
      "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19",
      "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26"))
    .setOutputCol("features")

  // Split the data into training and test sets (30% held out for testing)
  val Array(trainingData, testData) = finalData.randomSplit(Array(0.7, 0.3))

  // Train a DecisionTree model.
  val dt = new DecisionTreeClassifier()
    .setLabelCol("indexedLabel")
    .setFeaturesCol("features")

  // Convert indexed labels back to original labels.
  val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(labelIndexer.labels)

  // Chain indexers and tree in a Pipeline
  val pipeline = new Pipeline().setStages(Array(labelIndexer, indexerMedia, indexerAppOrSite, assembler, dt, labelConverter))

  // Train model.  This also runs the indexers.
  val model = pipeline.fit(trainingData)

  // Make predictions.
  val predictions = model.transform(testData)

  // Select example rows to display.
  predictions.select("predictedLabel", "label", "features").show(100)

  // Select (prediction, true label) and compute test error
  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  val accuracy = evaluator.evaluate(predictions)
  println("Test Error = " + (1.0 - accuracy))
  println("ACCURACY " + accuracy)

  spark.stop

}
