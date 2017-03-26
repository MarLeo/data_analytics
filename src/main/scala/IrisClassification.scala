import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession

/**
  * Created by marti on 26/03/2017.
  *
  * IrisClassification is a simple example of Classification using Apache Spark's machine learning pipeline
  */

object IrisClassification extends DataLoader{

  /**
    * Only expects a single arg
    * args(0) should have the path to the iris data train
    * args(1) should have the path to the iris data test
    */

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.master("local").appName("iris").getOrCreate()

    val train = args(0)
    val test = args(1)

    val training = loadIris(train, spark)
    val testing = loadIris(test, spark)

    // Keep features and y
    training.select("features", "y").show(20)
    testing.select("features", "y").show(20)

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
