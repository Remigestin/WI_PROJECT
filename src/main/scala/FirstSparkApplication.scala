import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession, functions}
import org.apache.spark.sql.functions._


import org.apache.hadoop.fs._;


object FirstSparkApplication extends App {

  predict()

  /**
   * From the model, it creates a csv with all the predictions
   *
   */
  def predict(): Unit = {

    val testDataName = "data-students.json"

    //Only for windows user I think
    System.setProperty("hadoop.home.dir", "C:\\winutils")

    val spark = SparkSession.builder
      .master("local[*]")
      .appName("FirstSparkApplication")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    // We retrieve the model
    val model = getModel(spark, testDataName)

    //test data cleaning
    val testData = spark.read.json(testDataName)
    val cleanedTest = Cleaner.cleanTestData(testData)
      .select(
        "bidfloor", "media", "appOrSite", "area",
        "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10",
        "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19",
        "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26")

    // Make predictions.
    val predictions = model.transform(cleanedTest)
      /*.select("predictedLabel", "bidfloor", "mediaIndex", "appOrSiteIndex", "area",
        "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10",
        "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19",
        "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26")*/

    val predictionsDF = predictions.select("predictedLabel")
    val finalColumn = predictionsDF.withColumn("label", predictionsDF("predictedLabel")).drop("predictedLabel")

    val newDf = testData.withColumn("id1", monotonically_increasing_id())
    val newPredictions = finalColumn.withColumn("id2", monotonically_increasing_id())

    val df2 = newDf
      .as("df1")
      .join(
        newPredictions.as("df2"),
        newDf("id1") === newPredictions("id2"),
       "inner"
      )
        .select(
          "df2.label",
          "df1.appOrSite",
          "df1.bidfloor",
          "df1.city",
          "df1.exchange",
          "df1.impid",
          "df1.interests",
          "df1.media",
          "df1.network",
          "df1.os",
          "df1.publisher",
          "df1.size",
          "df1.timestamp",
          "df1.type",
          "df1.user",
        )


    // Save all columns of test data
    val testDataForCsv = df2.withColumn("size", testData.col("size").cast("String"))

    saveInCsv(testDataForCsv)

    spark.stop
  }

  /**
   * Save the dataframe in a folder named "predictions_[day]_[time].csv"
   *
   * @param predictions : dataframe to save
   */
  private def saveInCsv(predictions: DataFrame): Unit = {
    val dir = new File("result").mkdir

    val date = LocalDateTime.now()
    val dateFormat = DateTimeFormatter.ofPattern("dd-MM-yyyy_HH-mm-ss")
    val formattedDate = dateFormat.format(date)

    predictions.coalesce(1)
      .write
      .mode("append")
      .option("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
      .option("header", "true")
      .format("csv")
      .save("result/predictions_" + formattedDate + ".csv")
  }

  /**
   *
   * @param spark            : spark session
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
        // Read json data
        val trainingData = spark.read.json(trainingDataName)
        val cleanedTrain = Cleaner.cleanTrainData(trainingData)
          .select(
            "label", "classWeightCol",
            "bidfloor", "media", "appOrSite", "area",
            "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10",
            "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19",
            "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26")

        // Index labels, adding metadata to the label column.
        val labelIndexer = new StringIndexer()
          .setInputCol("label")
          .setOutputCol("indexedLabel")
          .fit(cleanedTrain)

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
          .setInputCols(Array("bidfloor", "mediaIndex", "appOrSiteIndex", "area",
            "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10",
            "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19",
            "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26"))
          .setOutputCol("features")
          .setHandleInvalid("keep")

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
        val model = pipeline.fit(cleanedTrain)
        model.write.overwrite().save(modelPath)
        model
    }
  }

}
