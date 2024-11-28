/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


// Code by lathika - test
#include <iostream> // Required for std::cout
#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp> // For glm::isnan
#include <cmath> // For std::isnan



// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;

	// Code by lathika -  test
	glm::bvec3 row1NaN = glm::bvec3(glm::isnan(Vrk[0][0]), glm::isnan(Vrk[0][1]), glm::isnan(Vrk[0][2]));
	glm::bvec3 row2NaN = glm::bvec3(glm::isnan(Vrk[1][0]), glm::isnan(Vrk[1][1]), glm::isnan(Vrk[1][2]));
	glm::bvec3 row3NaN = glm::bvec3(glm::isnan(Vrk[2][0]), glm::isnan(Vrk[2][1]), glm::isnan(Vrk[2][2]));

	if (glm::any(row1NaN) || glm::any(row2NaN) || glm::any(row3NaN)) {
		printf("Vrk has NaN\n");
		return;
	}
	glm::bvec3 row1NaN1 = glm::bvec3(glm::isnan(T[0][0]), glm::isnan(T[0][1]), glm::isnan(T[0][2]));
	glm::bvec3 row2NaN1 = glm::bvec3(glm::isnan(T[1][0]), glm::isnan(T[1][1]), glm::isnan(T[1][2]));
	glm::bvec3 row3NaN1 = glm::bvec3(glm::isnan(T[2][0]), glm::isnan(T[2][1]), glm::isnan(T[2][2]));

	if (glm::any(row1NaN1) || glm::any(row2NaN1) || glm::any(row3NaN1)) {
		printf("T has NaN\n");
		return;
	}
	if (glm::any(glm::bvec1(glm::isnan(dL_da)))) {
    printf("dL_da has NaN\n");
	return;
	}
	if (glm::any(glm::bvec1(glm::isnan(dL_db)))) {
    printf("dL_db has NaN\n");
	return;
	}
	if (glm::any(glm::bvec1(glm::isnan(dL_dc)))) {
    printf("dL_dc has NaN\n");
	return;
	}

	if (glm::any(glm::bvec1(glm::isnan(dL_dT00)))) {
	printf("a: %.2f, b: %.2f, c: %.2f\n", a, b, c);
    printf("denom: %.2f, denom2inv: %.2f\n", denom, denom2inv);
    printf("dL_dconic_x: %.2f, dL_dconic_y: %.2f, dL_dconic_z: %.2f\n", dL_dconic.x, dL_dconic.y, dL_dconic.z);
    printf("dL_da: %.2f, dL_dc: %.2f, dL_db: %.2f\n", dL_da, dL_dc, dL_db);
    printf("T: %.2f %.2f %.2f %.2f %.2f %.2f\n", T[0][0], T[0][1], T[0][2], T[1][0], T[1][1], T[1][2]);
    printf("Vrk: %.2f %.2f %.2f %.2f %.2f %.2f\n", Vrk[0][0], Vrk[0][1], Vrk[0][2], Vrk[1][1], Vrk[1][2], Vrk[2][2]);
    printf("dL_dT00 has NaN\n");
	return;
	}
	if (glm::any(glm::bvec1(glm::isnan(dL_dT01)))) {
	printf("a: %.2f, b: %.2f, c: %.2f\n", a, b, c);
    printf("denom: %.2f, denom2inv: %.2f\n", denom, denom2inv);
    printf("dL_dconic_x: %.2f, dL_dconic_y: %.2f, dL_dconic_z: %.2f\n", dL_dconic.x, dL_dconic.y, dL_dconic.z);
    printf("dL_da: %.2f, dL_dc: %.2f, dL_db: %.2f\n", dL_da, dL_dc, dL_db);
    printf("T: %.2f %.2f %.2f %.2f %.2f %.2f\n", T[0][0], T[0][1], T[0][2], T[1][0], T[1][1], T[1][2]);
    printf("Vrk: %.2f %.2f %.2f %.2f %.2f %.2f\n", Vrk[0][0], Vrk[0][1], Vrk[0][2], Vrk[1][1], Vrk[1][2], Vrk[2][2]);
    printf("dL_dT01 has NaN\n");
	return;
	}
	if (glm::any(glm::bvec1(glm::isnan(dL_dT02)))) {
	printf("a: %.2f, b: %.2f, c: %.2f\n", a, b, c);
    printf("denom: %.2f, denom2inv: %.2f\n", denom, denom2inv);
    printf("dL_dconic_x: %.2f, dL_dconic_y: %.2f, dL_dconic_z: %.2f\n", dL_dconic.x, dL_dconic.y, dL_dconic.z);
    printf("dL_da: %.2f, dL_dc: %.2f, dL_db: %.2f\n", dL_da, dL_dc, dL_db);
    printf("T: %.2f %.2f %.2f %.2f %.2f %.2f\n", T[0][0], T[0][1], T[0][2], T[1][0], T[1][1], T[1][2]);
    printf("Vrk: %.2f %.2f %.2f %.2f %.2f %.2f\n", Vrk[0][0], Vrk[0][1], Vrk[0][2], Vrk[1][1], Vrk[1][2], Vrk[2][2]);
    printf("dL_dT02 has NaN\n");
	return;
	}
	if (glm::any(glm::bvec1(glm::isnan(dL_dT10)))) {
	printf("a: %.2f, b: %.2f, c: %.2f\n", a, b, c);
    printf("denom: %.2f, denom2inv: %.2f\n", denom, denom2inv);
    printf("dL_dconic_x: %.2f, dL_dconic_y: %.2f, dL_dconic_z: %.2f\n", dL_dconic.x, dL_dconic.y, dL_dconic.z);
    printf("dL_da: %.2f, dL_dc: %.2f, dL_db: %.2f\n", dL_da, dL_dc, dL_db);
    printf("T: %.2f %.2f %.2f %.2f %.2f %.2f\n", T[0][0], T[0][1], T[0][2], T[1][0], T[1][1], T[1][2]);
    printf("Vrk: %.2f %.2f %.2f %.2f %.2f %.2f\n", Vrk[0][0], Vrk[0][1], Vrk[0][2], Vrk[1][1], Vrk[1][2], Vrk[2][2]);
    printf("dL_dT10 has NaN\n");
	return;
	}
	if (glm::any(glm::bvec1(glm::isnan(dL_dT11)))) {
	printf("a: %.2f, b: %.2f, c: %.2f\n", a, b, c);
    printf("denom: %.2f, denom2inv: %.2f\n", denom, denom2inv);
    printf("dL_dconic_x: %.2f, dL_dconic_y: %.2f, dL_dconic_z: %.2f\n", dL_dconic.x, dL_dconic.y, dL_dconic.z);
    printf("dL_da: %.2f, dL_dc: %.2f, dL_db: %.2f\n", dL_da, dL_dc, dL_db);
    printf("T: %.2f %.2f %.2f %.2f %.2f %.2f\n", T[0][0], T[0][1], T[0][2], T[1][0], T[1][1], T[1][2]);
    printf("Vrk: %.2f %.2f %.2f %.2f %.2f %.2f\n", Vrk[0][0], Vrk[0][1], Vrk[0][2], Vrk[1][1], Vrk[1][2], Vrk[2][2]);
    printf("dL_dT11 has NaN\n");
	return;
	}
	if (glm::any(glm::bvec1(glm::isnan(dL_dT12)))) {
	printf("a: %.2f, b: %.2f, c: %.2f\n", a, b, c);
    printf("denom: %.2f, denom2inv: %.2f\n", denom, denom2inv);
    printf("dL_dconic_x: %.2f, dL_dconic_y: %.2f, dL_dconic_z: %.2f\n", dL_dconic.x, dL_dconic.y, dL_dconic.z);
    printf("dL_da: %.2f, dL_dc: %.2f, dL_db: %.2f\n", dL_da, dL_dc, dL_db);
    printf("T: %.2f %.2f %.2f %.2f %.2f %.2f\n", T[0][0], T[0][1], T[0][2], T[1][0], T[1][1], T[1][2]);
    printf("Vrk: %.2f %.2f %.2f %.2f %.2f %.2f\n", Vrk[0][0], Vrk[0][1], Vrk[0][2], Vrk[1][1], Vrk[1][2], Vrk[2][2]);
    printf("dL_dT12 has NaN\n");
	return;
	}

	if (glm::any(glm::bvec1(glm::isnan(dL_dJ00)))) {
    printf("dL_dJ00 has NaN\n");
	return;
	}
	if (glm::any(glm::bvec1(glm::isnan(dL_dJ02)))) {
    printf("dL_dJ02 has NaN\n");
	return;
	}
	if (glm::any(glm::bvec1(glm::isnan(dL_dJ11)))) {
    printf("dL_dJ11 has NaN\n");
	return;
	}
	if (glm::any(glm::bvec1(glm::isnan(dL_dJ12)))) {
    printf("dL_dJ12 has NaN\n");
	return;
	}
	
	glm::bvec3 isNaNVec_4 = glm::bvec3(glm::isnan(dL_dtx), glm::isnan(dL_dty), glm::isnan(dL_dtz));
	if (glm::any(isNaNVec_4)) {
    printf("dL_dt has NaN\n");
    return;
	}
	glm::bvec3 isNaNVec = glm::bvec3(glm::isnan(dL_dmean.x), glm::isnan(dL_dmean.y), glm::isnan(dL_dmean.z));
	if (glm::any(isNaNVec)) {
    printf("dL_dmean has NaN\n");
    return;
	}
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const float4* conic_opacity,	// Code by lathika
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// Code by lathika -  test
	if (glm::any(glm::isnan(dL_dmean))==1){
		printf( "dL_dmean has NaN: %d" , static_cast<int>(glm::any(glm::isnan(dL_dmean))) );
		return ;
	}

	// Code by lathika
	float4 con_o = conic_opacity[idx];

	// Compute gradient updates due to computing colors from SHs
	if (shs &&(con_o.w > 0.0f))	// Code by lathika - added ||(con_o.w > 0.0f) , only for positive gaussinas, mean will be affected by colors
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}


// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int P, 	// Code by lathika Test
	int W, int H,
	const float* __restrict__ bg_color,
	const bool* ctn_gauss_mask, // Code by lathika
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ depths,	// Code by lathika
	const float* __restrict__ look_at_var_arr,	// Code by lathika
	float* __restrict__ pos_dL_dAcummApha_arr, // Code by lathika
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	// Code by lathika
	// pos_dL_dAcummApha_arr :- This array will store the positive alpha values of the previouse positive gaussians.
														// If a positive gaussian was affected by negative gaussians then this accumulated alpha = pos_alpha + neg_accum_alpha,
														// where neg_accum_alpha = n_alpha_1 + n_alpha_2*(1+n_alpha_1) +  n_alpha_3*(1+n_alpha_2)*(1+n_alpha_1) + ........
	float T_final_neg = 1.0f;
	int pos_min_i = (pix.x + pix.y * W) * 1 * BLOCK_SIZE;	// Caution! Change if want, by considering size of pos_dL_dAcummApha_arr, defined just before launching this kernel
	int closest_pos_pointer = pos_min_i;	
	int closest_pos_glob_idx;
	int closest_pos_range_idx_ptr ;
	bool neg_gauss_loop = true;	// After finding a neg gaussian, there should be one loop. Neg gaussian loops should start after a another positive gaussian.
	float neg_alpha ;
	bool dlaa_p_once = false;	// dL_dAccumAlpha is stored in a cyclic manner, if a full cycle is completed, past data can be accessed by refering to the other end


	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_range_idx[BLOCK_SIZE]; // Code by lathika (again added volatile)

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			const int range_id =  range.y - progress - 1;		// Code by lathika
			collected_range_idx[block.thread_rank()] = range_id; // Code by lathika
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Code by lathika
			if (ctn_gauss_mask[collected_id[j]])
				continue;

			// Compute blending values, as before.
			float2 xy = collected_xy[j];	// Code by lathika removed const
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };	// Code by lathika removed const
			const float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;	// Code by lathika removed const
			if (power > 0.0f)
				continue;

			float G = exp(power);	// Code by lathika removed const

			// Code by lathika
			if (con_o.w < 0.0f)		// neg gauss detected
			{	
				int test_collected_range_idx =collected_range_idx[j];//Test  range.y - progress - 1
				int test_coll_id_1 = collected_id[j]; // Test

				neg_alpha = max(-0.99f, con_o.w * G);
				if (!neg_gauss_loop || neg_alpha > -1.0f / 255.0f)
					{continue;}
				neg_gauss_loop = false;
				
				float neg_depth;
				float neg_var;
				float pos_depth;
				float pos_var;
				float4 con_o_n;
				float4 con_o_p;
				int mpr = closest_pos_range_idx_ptr;	// max positive gauss index pointer - Maximum positive gauss that is affected by the neg gaussians
				int mpr_temp = closest_pos_range_idx_ptr;
				int mpr_idx = closest_pos_glob_idx;
				int n_idx = collected_id[j];	// Closest neg gauss index
				// Check If neg gauss is in the range
				pos_depth = depths[mpr_idx];
				pos_var = look_at_var_arr[mpr_idx];
				con_o_p = conic_opacity[mpr_idx]; 
				neg_depth = depths[n_idx];
				neg_var = look_at_var_arr[n_idx]; 
				if (abs(pos_depth-neg_depth) > std_diff_coff*(glm::sqrt(pos_var)+glm::sqrt(neg_var))||con_o_p.w < 0.0f)
				{
					continue;
				}
				int a=0; // test
				
				while (mpr_temp < range.y )	// To get the maximum positive gaussian that will be effected by this neg gauss set
				{	
					if (mpr_temp== range.y) // As a solution - might execute witrh errors if not
					{
						break;
					}

					//Test
					// if (mpr_temp>=range.y || mpr_temp<range.x)
					// 	{printf("mpr_temp = %d, range.y = %d,  range.x = %d ",mpr_temp, range.y, range.x);}

					mpr_idx = point_list[mpr_temp];

					// Test ------ Above clear
					//printf("mpr_temp = %d, range.y = %d,  range.x = %d , mpr_idx=%d",mpr_temp, range.y, range.x, mpr_idx);// Test
					
					//if (mpr_idx>=30314)//Test
						//printf("--mpr_idx = %d----",mpr_idx);
					//printf("---mpr_idx = %d----",mpr_idx);
					pos_depth = depths[mpr_idx];
					//printf("...pos_depth = %f, mpr_idx = %d...",pos_depth, mpr_idx);// Test
					pos_var = look_at_var_arr[mpr_idx];
					//printf("pos_var = %f",pos_var);// Test
					con_o_p = conic_opacity[mpr_idx]; 
					//printf("con_o_p.w = %f",con_o_p.w);// Test

					// mpr_temp++;// Test
					// continue;//Test
					if  (abs(pos_depth-neg_depth) > std_diff_coff*(glm::sqrt(pos_var)+glm::sqrt(neg_var)))
					{	//printf(" pos_depth ,  neg_depth , glm::sqrt(pos_var) ,  glm::sqrt(neg_var), count = %f, %f, %f, %f, %d \n",pos_depth, neg_depth,glm::sqrt(pos_var),glm::sqrt(neg_var), a);//Test
						break;
						}

					mpr = (con_o_p.w > 0.0f) ? mpr_temp: mpr;

					mpr_temp ++;

					// Test
					// if (a>100){	
					// 	printf("Loop_1_a = %d",a);
					// 	printf(" pos_depth ,  neg_depth , glm::sqrt(pos_var) ,  glm::sqrt(neg_var),dep_diff, std*coeff,std_diff_coff, count = %f, %f, %f, %f,%f,%f,%f, %d \n",pos_depth, neg_depth,glm::sqrt(pos_var),glm::sqrt(neg_var),abs(pos_depth-neg_depth),std_diff_coff*(glm::sqrt(pos_var)+glm::sqrt(neg_var)),std_diff_coff, a);//Test
					// }
					a++;// Test
				}
				//printf("loop pass a=%d ,mpr_temp = %d, range.y = %d,  range.y = %d , mpr_idx=%d",a,mpr_temp, range.y, range.x, mpr_idx);
				//continue;// Test - illegal mem access above

				int test_collected_range_idx_2 = collected_range_idx[j];//Test
				T_final_neg = 1.0f;;
				volatile int mlrp_n = collected_range_idx[j]; 				// main loop neg gauss range index pointer (Will be updated in the loop below) 
				int closest_neg_range_idx_ptr = collected_range_idx[j]; 	// Closest negative gaussian pointer
				pos_depth = depths[closest_pos_glob_idx];			// Closest pos gauss depth
				pos_var = look_at_var_arr[closest_pos_glob_idx];	// Closest pos gauss var
				int mlgi_n ;										// main loop neg gauss global index
				float dL_dAccumAlpha_p = 0.0f;
				int b = 0; // Test
				
				// test codes
				int test_collected_range_idx_3 = collected_range_idx[j];//Test
				int test_coll_id_2 = collected_id[j]; // Test
				int test_coll_id_3 = point_list[test_collected_range_idx_2]; // Test
				int test_coll_id_4 = point_list[mlrp_n]; // test
				int test_coll_id_5 = point_list[range.y - progress - 1]; // Test
				int test_coll_id_6 = point_list[closest_neg_range_idx_ptr]; // Test
				int test_coll_id_7 = point_list[test_collected_range_idx]; // Test
				//printf("test_coll_id_1 =%d, test_coll_id_2=%d, test_coll_id_3 =%d, test_coll_id_4=%d, test_coll_id_5 =%d",test_coll_id_1,test_coll_id_2,test_coll_id_3,test_coll_id_4,test_coll_id_5);
				//printf("mlrp_n out of bounds: %d, valid range: [%d, %d], collected_range_idx_fixed[%d] =%d, collected_range_idx[%d] =%d , test_coll_r_i =%d, test_coll_r_i_2 =%d, test_coll_r_i_3 =%d, range_id=%d\n",mlrp_n,  range.x, range.y,j,closest_neg_range_idx_ptr, j,collected_range_idx[j],test_collected_range_idx,test_collected_range_idx_2,test_collected_range_idx_3,range.y - progress - 1);//Test
				// Note:-
				//mlrp_n : 62327, valid range: [62282, 62409], collected_range_idx_fixed[81] =62327, collected_range_idx[81] =0 , test_coll_r_i =1089367776, test_coll_r_i_2 =62242, test_coll_r_i_3 =62327, range_id=62327
				// Here test_collected_range_idx_2 is reducing (...62243,62242,62241..)
				// mlrp_n, closest_neg_range_idx_ptr, test_collected_range_idx_3 and  range.y - progress - 1 remains at the same value (for a block i think) =62327
				// test_collected_range_idx which is, test_collected_range_idx = range.y - progress - 1; stays in a large value like 1089367776 (for ablock i think)
				// But this is not the case for pos gauss part, every where collected_id[j] gives the same value, 
				// in neg gaus also collected_id[j] stays the same ex- test_coll_id_1 =30095, test_coll_id_2=30095
				// test_coll_id_1 =30082, test_coll_id_2=30082, test_coll_id_3 =30082, test_coll_id_4=30082, test_coll_id_5 =13691
				// As for the conclusion here, using collected_range_idx[j] is better than range.y - progress - 1
				// But in the following loop
				// if (mlrp_n < range.x || mlrp_n >= range.y)
				// 	{
				// 		printf("mlrp_n out of bounds: %d, valid range: [%d, %d], collected_range_idx_fixed[%d] =%d, collected_range_idx[%d] =%d , test_coll_r_i =%d, test_coll_r_i_2 =%d, test_coll_r_i_3 =%d, range_id=%d\n",mlrp_n,  range.x, range.y,j,closest_neg_range_idx_ptr, j,collected_range_idx[j],test_collected_range_idx,test_collected_range_idx_2,test_collected_range_idx_3,range.y - progress - 1);
				// 		printf("test_coll_id_1 =%d, test_coll_id_2=%d, test_coll_id_3 =%d, test_coll_id_6=%d,test_coll_id_7=%d \n",test_coll_id_1,test_coll_id_2,test_coll_id_3,test_coll_id_6, test_coll_id_7);// Test
				// 		break; // Or handle error gracefully
				// 	}
				// This code gives,
				// mlrp_n out of bounds: -1, valid range: [0, 150], collected_range_idx_fixed[71] =78, collected_range_idx[71] =0 , test_coll_r_i =1079214080, test_coll_r_i_2 =78, test_coll_r_i_3 =78, range_id=78
				// test_coll_id_1 =19272, test_coll_id_2=19272, test_coll_id_3 =19272, test_coll_id_6=19272,test_coll_id_7=19272 
				// Here range.y - progress - 1, closest_neg_range_idx_ptr, test_collected_range_idx_2, test_collected_range_idx_3 gives the same vale 78
				// mlrp_n has gone out of range (because decrement issue)
				// test_collected_range_idx give 0 ???? (But in this test there were no illegal mem issue????)


				//Test illegal mem access in the below loop

				// Looping through neg gauss and getting the least affected neg gauss and calculating T_final_neg
				// And this will update mlrp_n
				while(mlrp_n >= range.x)
				{	
	
					if (mlrp_n < range.x || mlrp_n >= range.y)// Test - seems to be the solution 
					{
						break;
					}
					// Test
					// if (b>100){	
					// 	printf("Loop_2_b = %d",b);
					// 	printf(" pos_depth ,  neg_depth , glm::sqrt(pos_var) ,  glm::sqrt(neg_var), count = %f, %f, %f, %f, %d \n",pos_depth, neg_depth,glm::sqrt(pos_var),glm::sqrt(neg_var), b);//Test
					// }
					// b++;

					// Below is the original part but check the bellow test part also, there is an updated one
					// if  (abs(pos_depth-neg_depth) > std_diff_coff*(glm::sqrt(pos_var)+glm::sqrt(neg_var)) || con_o_n.w > 0.0f)
					// {
					// 	mlrp_n++;
					// 	//printf(" pos_depth ,  neg_depth , glm::sqrt(pos_var) ,  glm::sqrt(neg_var), count_a = %f, %f, %f, %f, %d \n",pos_depth, neg_depth,glm::sqrt(pos_var),glm::sqrt(neg_var), a);//Test
					// 	break;
					// }

					// Test
					if (mlrp_n < range.x || mlrp_n >= range.y)
					{
						printf("mlrp_n out of bounds: %d, valid range: [%d, %d], collected_range_idx_fixed[%d] =%d, collected_range_idx[%d] =%d , test_coll_r_i =%d, test_coll_r_i_2 =%d, test_coll_r_i_3 =%d, range_id=%d\n",mlrp_n,  range.x, range.y,j,closest_neg_range_idx_ptr, j,collected_range_idx[j],test_collected_range_idx,test_collected_range_idx_2,test_collected_range_idx_3,range.y - progress - 1);
						printf("test_coll_id_1 =%d, test_coll_id_2=%d, test_coll_id_3 =%d, test_coll_id_6=%d,test_coll_id_7=%d \n",test_coll_id_1,test_coll_id_2,test_coll_id_3,test_coll_id_6, test_coll_id_7);// Test
						break; // Or handle error gracefully
					}

					mlgi_n = point_list[mlrp_n];
					neg_depth = depths[mlgi_n];
					neg_var = look_at_var_arr[mlgi_n];
					con_o_n = conic_opacity[mlgi_n];
					xy = points_xy_image[mlgi_n];

					// Untile now ok
					
					// Test
					// if (point_list[mlrp_n] < 0 || point_list[mlrp_n] >= P)
					// {
					// 	printf("point_list[mlrp_n] out of bounds: %d\n", point_list[mlrp_n]);
					// 	break;
					// }
					// // above
					// if (depths == nullptr || look_at_var_arr == nullptr || conic_opacity == nullptr)
					// {
					// 	printf("Invalid pointer detected!\n");
					// 	return;
					// }
					// if (mlgi_n < 0 || mlgi_n >= P) // Replace `max_gaussians` with the valid size
					// {
					// 	printf("mlgi_n out of bounds: %d\n", mlgi_n);
					// 	break;
					// }
					
					
					
					
					// float sqrt_pos_var = (pos_var > 0) ? glm::sqrt(pos_var) : 0.0f;
					// float sqrt_neg_var = (neg_var > 0) ? glm::sqrt(neg_var) : 0.0f;
					// bellow
					//bool cond = abs(pos_depth - neg_depth) > std_diff_coff * (glm::sqrt(pos_var) + glm::sqrt(neg_var) ) ;
					
					if (abs(pos_depth - neg_depth) > std_diff_coff * (glm::sqrt(pos_var) + glm::sqrt(neg_var)))//|| con_o_n.w > 0.0f) // not aligned with forward pass, n1,n2,p1,n3,n4,p2 if n1 and p2 in the range, we concider the effect, but here we dont?
					{
						mlrp_n++;
						break;
					}

					// mlrp_n--;// Test
					// continue; // Test
					// Above
					mlrp_n--;

					d = { xy.x - pixf.x, xy.y - pixf.y };
					
					power = -0.5f * (con_o_n.x * d.x * d.x + con_o_n.z * d.y * d.y) - con_o_n.y * d.x * d.y;
					G = exp(power);
					neg_alpha = max(-0.99f, con_o.w * G);
					if (power > 0.0f || neg_alpha > -1.0f / 255.0f)
						{ 
							// if (mlrp_n > range.x) mlrp_n --; // Test
						continue;}
					T_final_neg = T_final_neg*(1 + neg_alpha);		
					
				}
				//continue;// Test issue above
				
				int lrp_p = closest_pos_range_idx_ptr; 	// loop pos gauss range index pointer
				int lrp_n = mlrp_n;						// loop neg gauss range index pointer
				int lgi_n;	// loop_glob_idx_negative
				int lgi_p;	// loop_glob_idx_positive
				int lr_pin_p = closest_pos_range_idx_ptr;	// loop range poisitive gaussian pin
				int dL_dalpha_n;
				float T_temp_n = T_final_neg;
				int dlaa_p = closest_pos_pointer - 1;	// Possitive dL_dAccumAlpha pointer (index)
				
				int c = 0; // Test
				
				// For each neg gauss (g0) starting from left (least) updating gradians of neg gauss starting from the closest neg gauss (to pos gauss) to it (g0), 
				// w.r.t the positive gaussians in it's (g0) range
				while(mlrp_n <= closest_neg_range_idx_ptr)
				{	
					// Test
					// if (c>1000){	
					// 	printf("Loop_3_c = %d",c);
					// 	printf("closest_neg_range_idx_ptr = %d, mlrp_n =%d",closest_neg_range_idx_ptr, mlrp_n);
					// }
					// c++;

					if (mlrp_n > closest_neg_range_idx_ptr)// For safety
					{
						break;
					}


					lrp_p = lr_pin_p;
					lrp_n = mlrp_n;
					mlgi_n = point_list[mlrp_n];
					mlrp_n ++;
					neg_depth = depths[mlgi_n];
					neg_var = look_at_var_arr[mlgi_n];
					xy = points_xy_image[mlgi_n];
					d = { xy.x - pixf.x, xy.y - pixf.y };
					con_o_n = conic_opacity[mlgi_n];
					power = -0.5f * (con_o_n.x * d.x * d.x + con_o_n.z * d.y * d.y) - con_o_n.y * d.x * d.y;
					G = exp(power);
					neg_alpha = max(-0.99f, con_o_n.w * G);
					if (power > 0.0f || neg_alpha > -1.0f / 255.0f)
						continue;
					
					int dd = 0; // Test
					// Socond loop to loop through positive gauss until the corresponding limit and mark the pin
					// mpr is the pos gauss pointer in range. This pos gauss is the last (with max distance) in the active range of the closest neg gauss (closest to pos gausses) 
					// When dlaa_p comming back from the cycle, it should not pass closest_pos_pointer again (going more than cycle otherwise)
					while (lrp_p <= mpr && dlaa_p != closest_pos_pointer)	
					{	
						// Test
						// if (dd>100){	
						// 	printf("Loop_4_d = %d",dd);
						// }
						// dd++;

						if (lrp_p > mpr || dlaa_p == closest_pos_pointer)	// For safety
						{
							break;
						}


						lgi_p = point_list[lrp_p];
						pos_depth = depths[lgi_p];
						pos_var = look_at_var_arr[lgi_p];
						con_o_p = conic_opacity[lgi_p];
						lrp_p ++;
						// If there were some neg gaus in middle if positive gauss range
						if (con_o_p.w < 0.0f){
							continue;}
						// If dlaa_p comes to the begining of the stored array, go to the right corresponding, corrner. (This can only be done once)
						if ((dlaa_p < pos_min_i) && dlaa_p_once)
						{
							dlaa_p = (pix.x + pix.y * W + 1) * 1 * BLOCK_SIZE -1;	
							if (dlaa_p >= H * W * 1 * BLOCK_SIZE || dlaa_p <0) // Test
								printf("closest_pos_pointer going out of range 2"); // Test
							dlaa_p_once = false;
							//printf("pos alpha range exceeded looping\n");// Test
						}
						else if ((dlaa_p < pos_min_i) && !dlaa_p_once){ // Else, we dont have the gradient info, so break.
							mlrp_n = closest_pos_range_idx_ptr;		  // Break the main loop
							//printf("pos alpha range not enough\n");// Test
							break;
						}

						if  (abs(pos_depth-neg_depth) > std_diff_coff*(glm::sqrt(pos_var)+glm::sqrt(neg_var)))
						{
							lr_pin_p = lrp_p-1;	// We are incrementing lrp_p before, so we need to reduce one.
							T_final_neg = T_final_neg/(1 + neg_alpha);
							break;
						}
						if (dlaa_p >= H * W * 1 * BLOCK_SIZE || dlaa_p<0) // Test
							printf("closest_pos_pointer going out of range 3"); // Test
						dL_dAccumAlpha_p = pos_dL_dAcummApha_arr[dlaa_p];	// This array is defined in the device. 
																			//It stored dL_dAccumAlpha for each poss gauss in each thread. (so uses a different index)
						dlaa_p--; 	// dL_dalpha (accum alpha if affected by neg gauss) is updated in the array only for the positive gaussians
						
						int e = 0; // Test
						// Grad update loop for neg gaussians
						while(lrp_n <= closest_neg_range_idx_ptr)
						{
							// Test
							// if (e>100){	
							// 	printf("Loop_5_e = %d",e);
							// }
							// e++;

							if (lrp_n > closest_neg_range_idx_ptr)// For safety
							{
								break;
							}

							lgi_n = point_list[lrp_n];
							lrp_n ++;
							xy = points_xy_image[lgi_n];
							d = { xy.x - pixf.x, xy.y - pixf.y };
							con_o_n = conic_opacity[lgi_n];
							power = -0.5f * (con_o_n.x * d.x * d.x + con_o_n.z * d.y * d.y) - con_o_n.y * d.x * d.y;
							G = exp(power);
							neg_alpha = max(-0.99f, con_o_n.w * G);
							if (power > 0.0f || neg_alpha > -1.0f / 255.0f)
								continue;
							T_temp_n = T_final_neg/(1 + neg_alpha);
							dL_dalpha_n = T_temp_n * dL_dAccumAlpha_p;

							// Helpful reusable temporary variables
							const float dL_dG = con_o_n.w * dL_dalpha_n;
							const float gdx = G * d.x;
							const float gdy = G * d.y;
							const float dG_ddelx = -gdx * con_o_n.x - gdy * con_o_n.y;
							const float dG_ddely = -gdy * con_o_n.z - gdx * con_o_n.y;
							// Update gradients w.r.t. 2D mean position of the Gaussian
							atomicAdd(&dL_dmean2D[lgi_n].x, dL_dG * dG_ddelx * ddelx_dx);
							atomicAdd(&dL_dmean2D[lgi_n].y, dL_dG * dG_ddely * ddely_dy);
							// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
							atomicAdd(&dL_dconic2D[lgi_n].x, -0.5f * gdx * d.x * dL_dG);
							atomicAdd(&dL_dconic2D[lgi_n].y, -0.5f * gdx * d.y * dL_dG);
							atomicAdd(&dL_dconic2D[lgi_n].w, -0.5f * gdy * d.y * dL_dG);
							// Update gradients w.r.t. opacity of the Gaussian
							atomicAdd(&(dL_dopacity[lgi_n]), G * dL_dalpha_n);
						}
					}
				}

				continue;
			}

			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);

			// Code by lathika
			if (closest_pos_pointer >= (pix.x + pix.y * W + 1) * 1 * BLOCK_SIZE )
			{
				closest_pos_pointer = pos_min_i ;
				dlaa_p_once = true;
			}
			if (closest_pos_pointer>= H * W * 1 * BLOCK_SIZE || closest_pos_pointer < 0)	// Test
				printf("closest_pos_pointer going out of range"); // Test
			pos_dL_dAcummApha_arr[closest_pos_pointer] = dL_dalpha;
			closest_pos_pointer ++ ;
			closest_pos_glob_idx = global_id;
			closest_pos_range_idx_ptr = collected_range_idx[j];
			neg_gauss_loop = true;
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const float4* conic_opacity,	// Code by lathika
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		conic_opacity,	// Code by lathika
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int P, // Code by lathika Test
	int W, int H,
	const float* bg_color,
	const bool* ctn_gauss_mask, // Code by lathika
	const float2* means2D,
	const float* depths,			// Code by lathika
	const float* look_at_var_arr, 	// Code by lathika
	const float4* conic_opacity,
	const float* colors,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors)
{
	// Code by lathika - to store possitive_dL_dalpha values
	float* pos_dL_dAcummApha_arr;
	int byte_size =  W * H * 1 * BLOCK_SIZE * sizeof(float) ;	// Change the size if want - search from "1 * BLOCK_SIZE"
	gpuErrorchk(cudaMalloc(&pos_dL_dAcummApha_arr, byte_size)); 
	gpuErrorchk(cudaMemset(pos_dL_dAcummApha_arr, 0, byte_size));	// initializing

	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		P, 	// Code by lathika Test
		W, H,
		bg_color,
		ctn_gauss_mask, // Code by lathika
		means2D,
		depths,				// Code by lathika
		look_at_var_arr,	// Code by lathika
		pos_dL_dAcummApha_arr, // Code by lathika
		conic_opacity,
		colors,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors
		);

	gpuErrorchk(cudaDeviceSynchronize()); // Code by lathika

	gpuErrorchk(cudaFree(pos_dL_dAcummApha_arr));
}