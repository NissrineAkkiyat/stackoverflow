package stackoverflow

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext.*
import org.apache.spark.rdd.RDD
import stackoverflow.StackOverflow.{clusterResults, kmeans, langSpread, vectorPostings}
import stackoverflow.StackOverflowSuite.sc

import java.io.File
import scala.io.{Codec, Source}
import scala.util.Properties.isWin

object StackOverflowSuite:
  val conf: SparkConf = new SparkConf().setMaster("local[*]").setAppName("StackOverflow")
  val sc: SparkContext = new SparkContext(conf)

class StackOverflowSuite extends munit.FunSuite:
  import StackOverflowSuite.*


  lazy val testObject = new StackOverflow {
    override val langs =
      List(
        "JavaScript", "Java", "PHP", "Python", "C#", "C++", "Ruby", "CSS",
        "Objective-C", "Perl", "Scala", "Haskell", "MATLAB", "Clojure", "Groovy")
    override def langSpread = 50000
    override def kmeansKernels = 45
    override def kmeansEta: Double = 20.0D
    override def kmeansMaxIterations = 120
  }

  lazy val stackOverflow: StackOverflow = testObject

  test("testObject can be instantiated") {
    val instantiatable = try {
      testObject
      true
    } catch {
      case _: Throwable => false
    }
    assert(instantiatable, "Can't instantiate a StackOverflow object")
  }


  import scala.concurrent.duration.given
  override val munitTimeout = 300.seconds

  test("groupedPostings should group questions and answers correctly") {
    val testData = List(
      Posting(1, 1, None, None, 10, Some("Scala")),
      Posting(1, 2, None, None, 20, Some("Java")),
      Posting(2, 3, Some(1), Some(1), 15, None),
      Posting(2, 4, Some(1), Some(1), 25, None),
      Posting(2, 5, Some(2), Some(2), 30, None)
    )
    val postingsRDD = sc.parallelize(testData)

    val stackOverflow = new StackOverflow()
    val grouped = stackOverflow.groupedPostings(postingsRDD)

    val expected = List(
      (1, Iterable((Posting(1, 1, None, None, 10, Some("Scala")), Posting(2, 3, Some(1), Some(1), 15, None)),
        (Posting(1, 1, None, None, 10, Some("Scala")), Posting(2, 4, Some(1), Some(1), 25, None)))),
      (2, Iterable((Posting(1, 2, None, None, 20, Some("Java")), Posting(2, 5, Some(2), Some(2), 30, None))))
    )

    // Convert expected and grouped to arrays, sort them, and then convert them back to lists
    val expectedSorted = expected.toArray
    val groupedSorted = grouped.collect().toList.toArray


    assert(groupedSorted.toList == expectedSorted.toList, "groupedPostings did not group questions and answers correctly")
  }

  test("groupedPostings should handle empty input RDD") {
    val emptyRDD = sc.parallelize(Seq.empty[Posting])

    val stackOverflow = new StackOverflow()
    val grouped = stackOverflow.groupedPostings(emptyRDD)

    assert(grouped.isEmpty(), "groupedPostings should return empty RDD when input RDD is empty")
  }

  test("groupedPostings should handle input RDD with only questions") {
    val testData = List(
      Posting(1, 1, None, None, 10, Some("Scala")),
      Posting(1, 2, None, None, 20, Some("Java"))
    )
    val postingsRDD = sc.parallelize(testData)

    val stackOverflow = new StackOverflow()
    val grouped = stackOverflow.groupedPostings(postingsRDD)

    assert(grouped.isEmpty(), "groupedPostings should return empty RDD when input RDD contains only questions")
  }

  test("groupedPostings should handle input RDD with only answers") {
    val testData = List(
      Posting(2, 3, Some(1), Some(1), 15, None),
      Posting(2, 4, Some(1), Some(1), 25, None),
      Posting(2, 5, Some(2), Some(2), 30, None)
    )
    val postingsRDD = sc.parallelize(testData)

    val stackOverflow = new StackOverflow()
    val grouped = stackOverflow.groupedPostings(postingsRDD)

    assert(grouped.isEmpty(), "groupedPostings should return empty RDD when input RDD contains only answers")
  }

  test("scoredPostings should compute high score for each question correctly") {
    val testData = List(
      (1, Iterable((Posting(1, 1, None, None, 10, Some("Scala")), Posting(2, 3, Some(1), Some(1), 15, None)),
        (Posting(1, 1, None, None, 10, Some("Scala")), Posting(2, 4, Some(1), Some(1), 25, None)))),
      (2, Iterable((Posting(1, 2, None, None, 20, Some("Java")), Posting(2, 5, Some(2), Some(2), 30, None))))
    )
    val grouped = sc.parallelize(testData)

    val stackOverflow = new StackOverflow()
    val scored = stackOverflow.scoredPostings(grouped)

    // Define expected high scores for each question
    val expectedScores = List((Posting(1, 1, None, None, 10, Some("Scala")), 25), (Posting(1, 2, None, None, 20, Some("Java")), 30))

    // Convert scored RDD to a list for easier comparison
    val scoredList = scored.collect().toList

    // Check if the scores for each question match the expected scores
    assert(scoredList == expectedScores, "scoredPostings did not compute high score for each question correctly")
  }

  test("vectorPostings should generate vectors with correct language indices") {
    // Créer des données de test pour les scores
    val testData = List(
      (Posting(1, 1, None, None, 10, Some("Scala")), 25),
      (Posting(1, 2, None, None, 20, Some("Java")), 30)
    )
    // Convertir les données de test en RDD
    val testDataRDD = sc.parallelize(testData)

    // Appeler la méthode vectorPostings avec les données de test
    val vectorizedRDD = vectorPostings(testDataRDD)

    // Récupérer les indices de langage à partir des vecteurs générés
    val langIndices = vectorizedRDD.map(_._1 / langSpread).distinct().collect()

    // Définir les résultats attendus
    val expectedLangIndices = Array(10, 1) // Indices pour scala et java dans l'ordre de la liste des langages

    // Vérifier si les indices de langage correspondent aux indices attendus
    assert(langIndices.sorted sameElements expectedLangIndices.sorted)
  }

  test("kmeans should cluster vectors correctly") {
    // Données de test pour les vecteurs
    val testData: List[(Int, Int)] = List(
      (1, 2), (2, 3), (3, 4), (8, 9), (9, 10), (10, 11)
    )

    // Créez un RDD à partir des données de test
    val vectors: RDD[(Int, Int)] = sc.parallelize(testData)

    // Définir les centres de cluster initiaux pour le test
    val initialMeans: Array[(Int, Int)] = Array(
      (1, 2), (8, 9)
    )

    // Appeler la méthode kmeans avec les données de test et les centres initiaux
    val resultMeans: Array[(Int, Int)] = kmeans(initialMeans, vectors)

    // Résultats attendus des centres de cluster
    val expectedMeans: Array[(Int, Int)] = Array(
      (2, 3), (9, 10)
    )

    // Vérifiez si les centres de cluster obtenus correspondent aux résultats attendus
    assert(resultMeans.sameElements(expectedMeans))
  }

  test("clusterResults should return correct results") {
    // Mock data for testing
    val mockMeans = Array((0, 50), (100000, 70), (200000, 60))
    val mockVectors = Array((0, 40), (100000, 60), (200000, 80),
      (100000, 50), (200000, 70), (0, 60),
      (0, 70), (100000, 80), (200000, 90),
      (0, 90), (100000, 100), (200000, 110))

    // Convert mock data to RDDs
    val meansRDD = sc.parallelize(mockMeans)
    val vectorsRDD = sc.parallelize(mockVectors)

    // Call the clusterResults method
    val results = clusterResults(meansRDD.collect(), vectorsRDD)

    // Print the results
    // println("Cluster Results:")
    // results.foreach(println)

    // Assertions
    assert(results.length == 3)
    val expectedLanguage = "Java"
    val expectedPercentage = 75.0
    val expectedClusterSize = 10
    val expectedMedianScore = 80
  }