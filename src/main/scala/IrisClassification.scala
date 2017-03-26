import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by marti on 26/03/2017.
  */

/**
  * IrisClassification is a simple example of Classification using Apache Spark's machine learning pipeline
  */

object IrisClassification extends DataLoader{

  /**
    * Only expects a single arg
    * arg(0) should have the path to the iris data
    */

  def main(args: Array[String]): Unit = {

    //val conf = new SparkConf(true).setAppName("iris").setMaster("local[*]")
    //val sc = new SparkContext(conf)
    val spark = SparkSession.builder.master("local").appName("iris").getOrCreate()

    val training = loadIris(args(0), spark)
    val testing = loadIris(args(1), spark)

    // Keep features and y
    training.select("features", "y")
    testing.select("features", "y")

    // print the result
    print("Training schema: ")
    training.printSchema()

    print("Training Data: ")
    print(training.show(20))

    // LOGISTIC REGRESSION TO TRAIN THE DATA
    val lr = new LogisticRegression()
              .setMaxIter(10)
              .setRegParam(0.03)
              .setElasticNetParam(0.8)
              .setFeaturesCol("features")
              .setLabelCol("y")

    // FIT THE MODEL
    val fit = lr.fit(training)

    // MAKE PREDICTION
    val prediction = fit.transform(testing)           //fit.transform(training)

    // SHOW PREDICTIONS
    prediction.select("y", "probability", "prediction").show(20)

    // PRINT THE COEFFICIENTS AND INTERCEPT FOR LOGISTIC REGRESSION
    print("Coefficients: " + fit.coefficientMatrix)
    print("Intercept: " + fit.intercept)

  }


}
