package net.finmath.finitedifference.utilities;

import java.util.function.DoubleBinaryOperator;

/**
 * Dependency-free plot descriptor for a two-dimensional finite-difference surface.
 *
 * <p>
 * The descriptor is compatible with plotting tools that consume a
 * {@link DoubleBinaryOperator}, for example external plot extensions, without
 * requiring this module to depend on those plotting libraries.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public final class FiniteDifferencePlotData2D {

    /**
     * The x min.
     */
    private final double xMin;
    /**
     * The x max.
     */
    private final double xMax;
    /**
     * The y min.
     */
    private final double yMin;
    /**
     * The y max.
     */
    private final double yMax;
    /**
     * The number of points x.
     */
    private final int numberOfPointsX;
    /**
     * The number of points y.
     */
    private final int numberOfPointsY;
    /**
     * The function.
     */
    private final DoubleBinaryOperator function;
    /**
     * The title.
     */
    private final String title;
    /**
     * The x axis label.
     */
    private final String xAxisLabel;
    /**
     * The y axis label.
     */
    private final String yAxisLabel;
    /**
     * The z axis label.
     */
    private final String zAxisLabel;

    /**
     * Creates a two-dimensional plot descriptor.
     *
     * @param xMin The lower x-axis bound.
     * @param xMax The upper x-axis bound.
     * @param yMin The lower y-axis bound.
     * @param yMax The upper y-axis bound.
     * @param numberOfPointsX The number of plot points in the x direction.
     * @param numberOfPointsY The number of plot points in the y direction.
     * @param function The function to plot.
     * @param title The plot title.
     * @param xAxisLabel The x-axis label.
     * @param yAxisLabel The y-axis label.
     * @param zAxisLabel The z-axis label.
     */
    public FiniteDifferencePlotData2D(
            final double xMin,
            final double xMax,
            final double yMin,
            final double yMax,
            final int numberOfPointsX,
            final int numberOfPointsY,
            final DoubleBinaryOperator function,
            final String title,
            final String xAxisLabel,
            final String yAxisLabel,
            final String zAxisLabel) {

        if (numberOfPointsX <= 0 || numberOfPointsY <= 0) {
            throw new IllegalArgumentException("The number of plot points must be positive.");
        }
        if (function == null) {
            throw new IllegalArgumentException("function must not be null.");
        }

        this.xMin = xMin;
        this.xMax = xMax;
        this.yMin = yMin;
        this.yMax = yMax;
        this.numberOfPointsX = numberOfPointsX;
        this.numberOfPointsY = numberOfPointsY;
        this.function = function;
        this.title = title;
        this.xAxisLabel = xAxisLabel;
        this.yAxisLabel = yAxisLabel;
        this.zAxisLabel = zAxisLabel;
    }

    public double getXMin() {
        return xMin;
    }

    public double getXMax() {
        return xMax;
    }

    public double getYMin() {
        return yMin;
    }

    public double getYMax() {
        return yMax;
    }

    public int getNumberOfPointsX() {
        return numberOfPointsX;
    }

    public int getNumberOfPointsY() {
        return numberOfPointsY;
    }

    public DoubleBinaryOperator getFunction() {
        return function;
    }

    public String getTitle() {
        return title;
    }

    public String getXAxisLabel() {
        return xAxisLabel;
    }

    public String getYAxisLabel() {
        return yAxisLabel;
    }

    public String getZAxisLabel() {
        return zAxisLabel;
    }
}
