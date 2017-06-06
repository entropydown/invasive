import org.scalatest.{FunSpec, Matchers}

class ExtremaSpec extends FunSpec with Matchers {
  describe("isExtrema") {
    it("should return true only if the provided pixel value is > its neighbors") {
      val input = Seq(0.0, 1, 1, 0)
      Extrema.isExtrema(input, 1.0) should equal(false)
      Extrema.isExtrema(input, 2) should equal(true)
    }

    it("should return false if the provided pixel value is < its neighbors") {
      val input = Seq(4, 5, 22.5, 3)
      Extrema.isExtrema(input, 2.8) should equal(true)
    }
  }
}