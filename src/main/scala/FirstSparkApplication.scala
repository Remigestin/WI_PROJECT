import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics, RegressionMetrics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.functions.udf


object FirstSparkApplication extends App{

  //Only for windows user I think
  System.setProperty("hadoop.home.dir", "C:\\winutils")

  val spark = SparkSession.builder
    .master("local[*]")
    .appName("FirstSparkApplication")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  // Read json data
  val jsonData = spark.read.json("data-students.json")

  jsonData.show(5)

  // DATA CLEANING
  val cleanedData = Cleaner.generalClean(jsonData)

  // Select only the needed columns
  val finalData = cleanedData.select(
    "label",
    "bidFloor", "media", "appOrSite","area",
    "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10",
    "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19",
    "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26")

  finalData.show(5)
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
    .setInputCols(Array("bidFloor", "mediaIndex", "appOrSiteIndex","area",
      "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10",
      "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19",
      "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26"))
    .setOutputCol("features")

  // Split the data into training and test sets (30% held out for testing)
  val Array(trainingData, testData) = finalData.randomSplit(Array(0.7, 0.3))


  val lr = new LogisticRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)
    .setLabelCol("indexedLabel")
    .setFeaturesCol("features")

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
  val pipeline = new Pipeline().setStages(Array(labelIndexer, indexerMedia, indexerAppOrSite, assembler, lr, labelConverter))

  // Train model.  This also runs the indexers.
  val model = pipeline.fit(trainingData)

  // Make predictions.
  val predictions = model.transform(testData)

  // Select (prediction, true label) and compute test error
  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  val accuracy = evaluator.evaluate(predictions)
  println("Test Error = " + (1.0 - accuracy))
  println("ACCURACY " + accuracy)




  val p = predictions.select("predictedLabel", "label", "features")

  val castToDouble = udf { label: String =>
    if (label == "false") 0.0
    else 1.0
  }

  val casted = castToDouble(p.col("label"))
  val newDataframe = p.withColumn("label", casted)

  val casted2 = castToDouble(p.col("predictedLabel"))
  val newDataframe2 = newDataframe.withColumn("predictedLabel", casted2)

  val rows: RDD[Row] = newDataframe2.rdd
  val predictionAndLabels = rows.map(row => (row.getAs[Double](0), row.getAs[Double](1)))

  // Instantiate metrics object
  val metrics = new MulticlassMetrics(predictionAndLabels)

  // Confusion matrix
  println("Confusion matrix:")
  println(metrics.confusionMatrix)

  // Overall Statistics
  val accuracy2 = metrics.accuracy
  println("Summary Statistics")
  println(s"Accuracy = " + accuracy2)

  // Precision by label
  val labels = metrics.labels
  labels.foreach { l =>
    println(s"Precision($l) = " + metrics.precision(l))
  }

  // Recall by label
  labels.foreach { l =>
    println(s"Recall($l) = " + metrics.recall(l))
  }

  // False positive rate by label
  labels.foreach { l =>
    println(s"FPR($l) = " + metrics.falsePositiveRate(l))
  }

  spark.stop

}
