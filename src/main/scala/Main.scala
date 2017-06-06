import org.opencv.core._
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

object Constants {
  val GAUSS_SIGMA_INITIAL = Math.sqrt(2)
  val GAUSS_PRE_BLUR = 1.0
  val SUGGESTED_ANTI_ALIAS = 0.5
  val MAX_OCTAVES = 4
  val MINIMUM_CONTRAST = 0.03

  val r = 10.0
  val CURVATURE_THRESHOLD = (r + 1) * (r + 1) / r
}

object OpenCvFunctional {
  def zeroMatFromInput(input: Mat) = Mat.zeros(input.size, input.`type`)

  def zeroMatFromInputHalfSize(input: Mat) = Mat.zeros(new Size(input.size.height / 2, input.size.width / 2), input.`type`)

  def gaussianBlur(input: Mat, sigma: Double) = {
    val output = zeroMatFromInput(input)

    // omfg, was specifying Core.border_type as sigma y value...
    Imgproc.GaussianBlur(input, output, new Size(0, 0), sigma, sigma)
    output
  }

  def halfSize(input: Mat) = {
    val output = zeroMatFromInputHalfSize(input)

    Imgproc.resize(input, output, new Size(0, 0), 0.5, 0.5, Imgproc.INTER_NEAREST)

    output
  }

  def doubleSize(input: Mat) = {
    val output = Mat.zeros(new Size(input.size.height * 2, input.size.width * 2), input.`type`)

    Imgproc.resize(input, output, new Size(0, 0), 2, 2, Imgproc.INTER_LINEAR)

    output
  }

  def subtract(a: Mat, b: Mat) = {
    val output = zeroMatFromInput(a)

    Core.subtract(a, b, output, new Mat(), CvType.CV_32FC1)

    output
  }
}

object Main extends App {
  import Constants._
  import OpenCvFunctional._
  import Extrema._

  System.loadLibrary(Core.NATIVE_LIBRARY_NAME)

  val original = {
    val read = Imgcodecs.imread("/Users/entropy/sift/damo.jpg")
    val output = Mat.zeros(read.size, CvType.CV_8UC1)
    Imgproc.cvtColor(read, output, Imgproc.COLOR_BGR2GRAY)

    val fpImage = Mat.zeros(output.size, CvType.CV_32FC1)

    // critical point: the algorithm assumes the intensity values are (0, 1]
    output.convertTo(fpImage, CvType.CV_32FC1, 1, 0)

    val preBlured = gaussianBlur(fpImage, SUGGESTED_ANTI_ALIAS)

    val doubleSized = doubleSize(preBlured)

    gaussianBlur(doubleSized, GAUSS_PRE_BLUR)
  }

  /**
    * remember comparisons are made with images from the same octave, so the sizes of top, mid and bottom will be the same
    * if it is an extrema (lowest or highest value in its neighbourhood, set the output Mat position as 255
    * @param top
    * @param middle
    * @param bottom
    * @return A Mat of extrema
    */
  def localExtremaDetection(top: Mat, middle: Mat, bottom: Mat): Mat = {
    val yMax = middle.size.height.toInt-2
    val xMax = middle.size.width.toInt-2

    val output = zeroMatFromInput(middle)

    // skip the edges
    (1 to yMax).map { y =>
      (1 to xMax).map { x =>
        val current = middle.get(y, x).head

        if (Math.abs(current) > MINIMUM_CONTRAST) {
          val pixelsForExtremaSearch =
            neighbourhoodPixelSearch(top, y, x, considerStartingPoint = true) ++
            neighbourhoodPixelSearch(middle, y, x, considerStartingPoint = false) ++
            neighbourhoodPixelSearch(bottom, y, x, considerStartingPoint = true)

          if (isExtrema(pixelsForExtremaSearch, current)) { //&& passesEdgeCurvatureDetection(middle, y, x)) {
            output.put(y, x, 255)
          }
        }
      }
    }

    output
  }

  // 4 octaves
  (1 to MAX_OCTAVES).foldLeft(original) { case (octaveImage, octaveNumber) =>
    println(s"====== octave $octaveNumber ======")

    var sigma = GAUSS_SIGMA_INITIAL * 0.5 * Math.pow(2, octaveNumber - 1)

    // 5 blurs per octave
    println("generating gauss pyramid")
    val blurredImages = (0 to 4) map { i =>
      //println(s"sigma: $sigma")
      val output = gaussianBlur(octaveImage, sigma)
      sigma = Math.pow(2, 1.0/2) * sigma
      Imgcodecs.imwrite(s"/Users/entropy/sift/damo-blur-octave-$octaveNumber-blur-$i.jpg", output)
      output
    }

    // four of these
    println("computing dog")
    val differenceOfGaussians = blurredImages.sliding(2, 1).map { case groupOfTwo =>
      val img1 = groupOfTwo.head
      val img2 = groupOfTwo.last

      subtract(img1, img2)
    }.toList

    differenceOfGaussians.zipWithIndex.foreach { case (img, i) =>
      Imgcodecs.imwrite(s"/Users/entropy/sift/damo-dog-octave-$octaveNumber-$i.jpg", img)
    }

    // two of these
    println("detecting extrema")
    val extrema = differenceOfGaussians.sliding(3, 1).map { scalespaceChunk =>
      val top = scalespaceChunk(0)
      val middle = scalespaceChunk(1)
      val bottom = scalespaceChunk(2)

      localExtremaDetection(top, middle, bottom)
    }

    extrema.zipWithIndex.foreach { case (img, i) =>
      Imgcodecs.imwrite(s"/Users/entropy/sift/damo-extrema-octave-$octaveNumber-$i.jpg", img)

      val oi = octaveImage.clone()
      var keypoints = 0

      (0 to oi.size.height.toInt - 1).map { y =>
        (0 to oi.size.width.toInt - 1).map { x =>
          val keypoint = img.get(y, x).head

          if (keypoint > 0) {
            keypoints += 1
            oi.put(y, x, 255)
          }
        }
      }

      println(s"extrema $i: detected $keypoints")

      // overlay extrema detection on top of octave images
      Imgcodecs.imwrite(s"/Users/entropy/sift/damo-extrema-overlay-octave-$octaveNumber-$i.jpg", oi)
    }

    if (octaveNumber < MAX_OCTAVES) {
      println("scaling image")
      halfSize(octaveImage)
    } else {
      octaveImage
    }
  }
}