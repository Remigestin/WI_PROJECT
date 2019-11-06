import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.udf


object FirstSparkApplication extends App {

  predict("data-students.json")

  /**
   * From the model, it creates a csv with all the predictions
   * @param testDataName : data on which prediction is made
   */
  def predict(testDataName: String): Unit = {

    //Only for windows user I think
    System.setProperty("hadoop.home.dir", "C:\\winutils")

    val spark = SparkSession.builder
      .master("local[*]")
      .appName("FirstSparkApplication")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    // We retrieve the model
    val model = getModel(spark, "data-students.json")

    // Split the data into training and test sets (30% held out for testing)
    val testData = prepareData(spark, testDataName).randomSplit(Array(0.7, 0.3))(1)

    // Make predictions.
    val predictions = model.transform(testData)
    predictions.select("predictedLabel", "label", "features").show(100)

    buildMetrix(predictions)
    spark.stop
  }

  /**
   * Print in the console all the metrix of the predictions in parameter
   * @param predictions : dataFrame of the predictions
   */
  private def buildMetrix(predictions: DataFrame): Unit = {
    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))
    println("ACCURACY " + accuracy)


    val p = predictions.select("predictedLabel", "label", "features")

    // Code a nettoyer
    // Cast les colonnes label et predictedLabel en Double pour les metrics
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
  }

  /**
   *
   * @param spark : spark session
   * @param trainingDataName : name of the training dataFrame
   * @return : The model loaded if already exists, otherwise the model created.
   */
  private def getModel(spark: SparkSession, trainingDataName: String): PipelineModel = {
    val modelPath = "./spark-lr-model"
    try {
      //if model exists
      PipelineModel.load(modelPath)
    } catch {
      //else build the model
      case e: Exception =>
        // We prepare data to train the model
        val trainingData = prepareData(spark, trainingDataName)
        // Index labels, adding metadata to the label column.
        val labelIndexer = new StringIndexer()
          .setInputCol("label")
          .setOutputCol("indexedLabel")
          .fit(trainingData)

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
          .setInputCols(Array("bidFloor", "mediaIndex", "appOrSiteIndex", "area",
            "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10",
            "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19",
            "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26"))
          .setOutputCol("features")

        val lr = new LogisticRegression()
          .setWeightCol("classWeightCol")
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
        model.write.overwrite().save(modelPath)
        model
    }
  }

  /**
   *
   * @param spark : spark session
   * @param data : name of the dataFrame to prepare
   * @return : the dataFrame prepared
   */
  private def prepareData(spark: SparkSession, data: String): DataFrame = {
    // Read json data
    val jsonData = spark.read.json(data)

    // DATA CLEANING
    val cleanedData = Cleaner.generalClean(jsonData)

    // Select only the needed columns
    cleanedData.select(
      "label", "classWeightCol",
      "bidFloor", "media", "appOrSite", "area",
      "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10",
      "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19",
      "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26")

  }

}
