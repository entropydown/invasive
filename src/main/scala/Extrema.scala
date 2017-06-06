import org.opencv.core.Mat

object Extrema {
  import Constants._

  /**
    * The provided startingPosition is assumed to be a position where the search will not step out-of-bounds
    * @param image
    * @param y
    * @param x
    * @param considerStartingPoint Whether the starting point is considered in extrema detection
    * @return The extrema in the neighborhood surrounding the given starting coordinates
    */
  def neighbourhoodPixelSearch(image: Mat, y: Int, x: Int, considerStartingPoint: Boolean = false): Seq[Double] =
    Seq(
      image.get(y-1, x-1).head,
      image.get(y, x-1).head,
      image.get(y+1, x-1).head,

      image.get(y-1, x).head,
      image.get(y+1, x).head,

      image.get(y-1, x+1).head,
      image.get(y, x+1).head,
      image.get(y+1, x+1).head
    ) ++ { if (considerStartingPoint) Seq(image.get(y, x).head) else Seq.empty[Double] }

  def isExtrema(neighborPixelValues: Seq[Double], currentPixelValue: Double): Boolean =
    if (neighborPixelValues.forall(currentPixelValue > _))
      true
    else if (neighborPixelValues.forall(currentPixelValue < _))
      true
    else
      false
}