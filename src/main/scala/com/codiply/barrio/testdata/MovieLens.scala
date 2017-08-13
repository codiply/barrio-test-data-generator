package com.codiply.barrio.testdata

import scala.math.sqrt

import java.nio.file.Paths
import java.nio.file.Files
import java.nio.file.StandardOpenOption
import java.nio.charset.StandardCharsets

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.recommendation.ALSModel
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types._
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object MovieLens {
  def main(args: Array[String]) {
    run()
  }

  def run(): Unit = {
    Logger.getRootLogger.setLevel(Level.ERROR)

    val numberOfPartitions = 16
    val numberOfOutputs = 4
    val inputDataFolder = "./data/ml-latest/"
    val maxIterations = 10
    val ranks = List(16, 32, 64, 128, 256)
    val lambdas = List(0.01, 0.1, 1.0)
    val rmseOutputFile = "./data/ml-als-rmse.txt"

    val conf = new SparkConf()
      .setAppName("MovieLens")
      .setMaster("local[4]")
      .set("spark.driver.memory", "4g")

    val sc = new SparkContext(conf)
    val spark = new SQLContext(sc)

    import spark.implicits._

    val movies = loadMovies(spark, inputDataFolder + "movies.csv").cache()
    val links = loadLinks(spark, inputDataFolder + "links.csv").cache()

    movies.createOrReplaceTempView("movies")
    links.createOrReplaceTempView("links")

    val ratings = loadRatings(spark, inputDataFolder + "ratings.csv").repartition(numberOfPartitions).cache()

    Files.write(Paths.get(rmseOutputFile), "Errors\n".getBytes(StandardCharsets.UTF_8), StandardOpenOption.CREATE)

    ranks.foreach(rank => {
      val model = fitBestModel(spark, ratings, rank, maxIterations, lambdas, rmseOutputFile)

      model.itemFactors.createOrReplaceTempView("itemFactors")

      val results = spark.sql("""
        SELECT lnk.imdbId AS id,
               fct.features AS features,
               mv.title AS title
        FROM itemFactors as fct
        JOIN movies AS mv
          ON fct.id = mv.movieId
        JOIN links as lnk
          ON fct.id = lnk.movieId
        """)
        .as[(String, Array[Double], String)].map { case (id, features, title) =>
          (id + "@~@" + features.mkString(",")) + "@~@" + title }.toDF("line").cache()

      val outputFolder = s"./data/ml-item-factors/rank-$rank"
      results.coalesce(numberOfOutputs).write.text(outputFolder)
      results.coalesce(1).write.text(outputFolder + "-single")
    })

    sc.stop()
  }

  def loadCsv(spark: SQLContext, schema: StructType, file: String): Dataset[Row] =
    spark.read
      .schema(schema)
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .load(file)

  def loadRatings(spark: SQLContext, file: String): Dataset[Row] = {
    val schema = StructType(Array(
        StructField("userId", LongType, false),
        StructField("movieId", LongType, false),
        StructField("rating", FloatType, false),
        StructField("timestamp", LongType, false)))
    loadCsv(spark, schema, file)
  }

  def loadMovies(spark: SQLContext, file: String): Dataset[Row] = {
    val schema = StructType(Array(
        StructField("movieId", LongType, false),
        StructField("title", StringType, false),
        StructField("genres", StringType, false)))
    loadCsv(spark, schema, file)
  }

  def loadLinks(spark: SQLContext, file: String): Dataset[Row] = {
    val schema = StructType(Array(
        StructField("movieId", LongType, false),
        StructField("imdbId", StringType, false),
        StructField("tmdbId", StringType, false)))
    loadCsv(spark, schema, file)
  }

  def fitBestModel(
      spark: SQLContext,
      ratings: Dataset[Row],
      rank: Int,
      maxIterations: Int,
      lambdas: List[Double],
      rmseOutputFile: String): ALSModel = {
    val splits = ratings.randomSplit(Array(0.8, 0.2))
    val train = splits(0).cache()
    val test = splits(1).cache()

    val modelForLambda = (lambda: Double) =>
      new ALS()
        .setMaxIter(maxIterations)
        .setRegParam(lambda)
        .setRank(rank)
        .setUserCol("userId")
        .setItemCol("movieId")
        .setRatingCol("rating")

    val bestLambda = lambdas.map(lambda => {
      val model = modelForLambda(lambda).fit(train)
      val predictions = model.transform(test)
      val error = computeRmse(spark, predictions)
      Files.write(Paths.get(rmseOutputFile),
          s"Rank: $rank lambda: $lambda RMSE: $error\n".getBytes(StandardCharsets.UTF_8),
          StandardOpenOption.APPEND)
      println()
      (lambda, error)
    }).minBy(_._2)._1

    Files.write(Paths.get(rmseOutputFile),
          s"Rank: $rank best lambda chosen: $bestLambda\n".getBytes(StandardCharsets.UTF_8),
          StandardOpenOption.APPEND)

    modelForLambda(bestLambda).fit(ratings)
  }

  def computeRmse(spark: SQLContext, predictions: Dataset[Row]): Double = {
    predictions.createOrReplaceTempView("predictions")

    val meanSquareError = spark.sql("""
      SELECT AVG((rating - prediction) * (rating - prediction))
      FROM predictions
      WHERE NOT ISNAN(prediction)
      """)
    sqrt(meanSquareError.first().getAs[Double](0))
  }
}
