import Dependencies._

ThisBuild / scalaVersion := "2.13.4"
ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / organization := "io.github.novakov-alexey"
ThisBuild / organizationName := "novakov-alexey"

lazy val root = (project in file("."))
  .settings(
    name := "tensorflow-scala-example",
    libraryDependencies ++= Seq(
      "org.platanios" %% "tensorflow-data" % tensorFlowScalaVer,
      "org.platanios" %% "tensorflow" % tensorFlowScalaVer classifier "darwin"
    )
  )
