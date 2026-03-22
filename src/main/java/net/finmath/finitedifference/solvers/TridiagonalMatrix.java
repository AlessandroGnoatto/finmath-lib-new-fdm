package net.finmath.finitedifference.solvers;

public class TridiagonalMatrix {

	public final double[] lower;
	public final double[] diag;
	public final double[] upper;

	public TridiagonalMatrix(final int n) {
		this.lower = new double[n];
		this.diag = new double[n];
		this.upper = new double[n];
	}
}