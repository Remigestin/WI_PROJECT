import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.udf

object Cleaner {

  /**
   * Complete cleaning of the training dataframe
   *
   * @param df : dataframe containing a colomn "interests" and "size"
   * @return the given dataframe cleaned
   */
  def cleanTrainData(df: DataFrame): DataFrame = {

    /** Delete rows with a null value in the columns of features */
    val dfCleaned_1 = df.na.drop("any", Seq("bidfloor", "appOrSite", "media", "interests", "size"))

    /** Clean Interests Column */
    val dfCleaned_2 = cleanInterests(dfCleaned_1)

    /** Add column with ratio */
    val dfCleaned_3 = addClassWeightCol(dfCleaned_2)

    /** Cast labels to String */
    val dfCleaned_4 = dfCleaned_3.withColumn("label", df.col("label").cast("String"))

    /** Clean Size Column */
    cleanSize(dfCleaned_4)
  }

  /**
   * Complete cleaning of the test dataframe
   *
   * @param df : dataframe containing a colomn "interests" and "size"
   * @return the given dataframe cleaned
   */
  def cleanTestData(df: DataFrame): DataFrame = {

    /** Clean Interests Column */
    val dfCleaned_1 = cleanInterests(df)

    /** Clean Size Column */
    cleanSize(dfCleaned_1)

  }

  /**
   * Clean "interests" column
   *
   * @param df : dataframe containing a colomn "interests"
   * @return the given dataframe updated with new interests columnslabel
   */
  def cleanInterests(df: DataFrame): DataFrame = {

    /** REPLACE NULL WITH N/A */
    /** KEEP ONLY INTERESTS IN THE RIGHT FORMAT (code starting with IAB) */
    val removeIfNotCodeUDF = udf { interests: String =>
      interests.split(",").filter(_.startsWith("IAB")) mkString ","
    }

    val interestsCleaned = removeIfNotCodeUDF(df.col("interests"))

    val newDataframe = df.withColumn("interests", interestsCleaned)

    /** CREATE 26 INTEREST COLUMNS */
    createInterestColumns(newDataframe)
  }


  /**
   * Create 26 columns for each interest class.
   * Each column contains 1 if the row has the interest specified in "interests" column, else contains 0
   *
   * @param df : dataframe containing a column "interests"
   * @return the given dataframe updated with new columns for each interest
   */
  def createInterestColumns(df: DataFrame): DataFrame = {

    @scala.annotation.tailrec
    def loop(accDataframe: DataFrame, interestNb: Int): DataFrame = {

      if (interestNb == 27) accDataframe

      else {
        val IABn = "IAB" + interestNb

        val containsInterestUDF = udf { listInterests: String =>
          if (listInterests.contains(IABn + ",") || listInterests.contains(IABn + "-")) 1
          else 0
        }

        val IABnColumn = containsInterestUDF(accDataframe.col("interests"))
        val dfUpdated = accDataframe.withColumn(IABn, IABnColumn)

        loop(dfUpdated, interestNb + 1)
      }
    }

    loop(df, 1)
  }

  /**
   *
   * @param df : dataframe containing a column "size"
   * @return : the given dataframe updated with a new column area (= 0 if null, 1 if small size, 2 if medium, 3 if big)
   */
  def cleanSize(df: DataFrame): DataFrame = {

    val sizeArea = udf { size: Seq[Long] =>

      val area = size(0) * size(1)
      //big area
      if (area > 150000) 3
      else {
        //medium area
        if (area > 60000) 2
        //small area
        else 1
      }
    }

    val columnSize = sizeArea(df.col("size"))
    val dfUpdated = df.withColumn("area", columnSize)
    dfUpdated
  }

  /**
   *
   * @param df : a non balanced dataFrame
   * @return : a dataFrame with a column to balance the part of false and true labels
   */
  def addClassWeightCol(df: DataFrame): DataFrame = {

    val addRatio = udf { label: Boolean =>
      val balanceRatio = 0.97509967
      if (label) balanceRatio
      else 1 - balanceRatio
    }

    val labelWithRadio = addRatio(df.col("label"))
    df.withColumn("classWeightCol", labelWithRadio)
  }
}