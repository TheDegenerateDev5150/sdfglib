#pragma once

#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace passes {
namespace normalization {

/**
 * Normalize a StructuredSDFG in place.
 *
 * Runs the loop-normalization pipeline. When `optimize_kernel_size` is set, also performs
 * map fusion before and after loop normalization to optimize kernel dimensions.
 *
 * This function performs:
 * 1. (Optional) Initial map fusion without init-into-reduction hoisting
 * 2. Loop distribution and stride minimization
 * 3. (Optional) Final map fusion with init-into-reduction hoisting
 */
void normalize_sdfg(sdfg::StructuredSDFG& sdfg, bool optimize_kernel_size = true);

} // namespace normalization
} // namespace passes
} // namespace sdfg
