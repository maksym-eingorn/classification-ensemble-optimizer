// Copyright 2026 Maksym Eingorn
// SPDX-License-Identifier: Apache-2.0

// native/triplet_search.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <omp.h>
#include <tuple>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

namespace {

using InputArray = py::array_t<
    double,
    py::array::c_style | py::array::forcecast
>;

// RAII guard: temporarily set OpenMP threads and restore on scope exit
struct OmpThreadGuard {
    int prev_threads;
    bool active;

    explicit OmpThreadGuard(int n_threads)
        : prev_threads(omp_get_max_threads()), active(false) {
        if (n_threads > 0) {
            omp_set_num_threads(n_threads);
            active = true;
        }
    }

    ~OmpThreadGuard() {
        if (active) {
            omp_set_num_threads(prev_threads);
        }
    }
};


struct TripletSolution {
    long double sse;
    double wi;
    double wj;
    double wk;
};


struct TripletCandidate {
    double brier_score;
    int i;
    int j;
    int k;
    double wi;
    double wj;
    double wk;
};


/*
 * Strict total ordering for retained triplet candidates.
 *
 * Lower Brier score is better. For exactly equal scores, lower model-column
 * indices are preferred lexicographically. Used both by the top-N heaps and
 * the final ascending sort, making equal-score ordering deterministic.
 *
 * std::push_heap with this comparator places the worst retained candidate
 * at heap.front().
 */
struct TripletCandidateLess {
    bool operator()(
        const TripletCandidate& lhs,
        const TripletCandidate& rhs
    ) const {
        if (lhs.brier_score != rhs.brier_score) {
            return lhs.brier_score < rhs.brier_score;
        }

        if (lhs.i != rhs.i) {
            return lhs.i < rhs.i;
        }

        if (lhs.j != rhs.j) {
            return lhs.j < rhs.j;
        }

        return lhs.k < rhs.k;
    }
};


inline long double compute_triplet_sse(
    double wi,
    double wj,
    double wk,
    double gii,
    double gij,
    double gik,
    double gjj,
    double gjk,
    double gkk,
    double si,
    double sj,
    double sk,
    double y2
) {
    const long double lwi = static_cast<long double>(wi);
    const long double lwj = static_cast<long double>(wj);
    const long double lwk = static_cast<long double>(wk);

    const long double lgii = static_cast<long double>(gii);
    const long double lgij = static_cast<long double>(gij);
    const long double lgik = static_cast<long double>(gik);
    const long double lgjj = static_cast<long double>(gjj);
    const long double lgjk = static_cast<long double>(gjk);
    const long double lgkk = static_cast<long double>(gkk);

    const long double lsi = static_cast<long double>(si);
    const long double lsj = static_cast<long double>(sj);
    const long double lsk = static_cast<long double>(sk);
    const long double ly2 = static_cast<long double>(y2);

    const long double linear = lwi * lsi + lwj * lsj + lwk * lsk;

    const long double quadratic =
        lwi * lwi * lgii +
        lwj * lwj * lgjj +
        lwk * lwk * lgkk +
        2.0L * lwi * lwj * lgij +
        2.0L * lwi * lwk * lgik +
        2.0L * lwj * lwk * lgjk;

    long double sse = ly2 - 2.0L * linear + quadratic;

    /*
     * Mathematically SSE is nonnegative. Tiny negative values can arise from
     * floating-point cancellation in the Gram-matrix representation.
     */
    if (sse < 0.0L) {
        sse = 0.0L;
    }

    return sse;
}


/*
 * Solve
 *
 *     min ||wi * pi + wj * pj + wk * pk - y||^2
 *
 * subject to
 *
 *     wi >= 0
 *     wj >= 0
 *     wk >= 0
 *     wi + wj + wk = 1.
 *
 * This is a convex quadratic program on a two-dimensional simplex.
 *
 * For three weights, its exact active-set solution requires checking:
 *
 *   1. the three simplex vertices;
 *   2. the optimum on each of the three simplex edges;
 *   3. the stationary point in the simplex interior.
 *
 * If the two-dimensional interior quadratic is singular, a global optimum
 * can always be represented on the boundary, so the edge/vertex checks are
 * sufficient.
 */
inline TripletSolution solve_convex_triplet(
    double gii,
    double gij,
    double gik,
    double gjj,
    double gjk,
    double gkk,
    double si,
    double sj,
    double sk,
    double y2
) {
    TripletSolution best{
        std::numeric_limits<long double>::infinity(),
        1.0,
        0.0,
        0.0
    };

    auto consider = [&](double wi, double wj, double wk) {
        const long double sse = compute_triplet_sse(
            wi,
            wj,
            wk,
            gii,
            gij,
            gik,
            gjj,
            gjk,
            gkk,
            si,
            sj,
            sk,
            y2
        );

        if (sse < best.sse) {
            best.sse = sse;
            best.wi = wi;
            best.wj = wj;
            best.wk = wk;
        }
    };

    /*
     * Vertices.
     *
     * Testing vertices explicitly also gives deterministic behavior when
     * duplicate model columns make the convex optimum non-unique.
     */
    consider(1.0, 0.0, 0.0);
    consider(0.0, 1.0, 0.0);
    consider(0.0, 0.0, 1.0);

    const long double lgii = static_cast<long double>(gii);
    const long double lgij = static_cast<long double>(gij);
    const long double lgik = static_cast<long double>(gik);
    const long double lgjj = static_cast<long double>(gjj);
    const long double lgjk = static_cast<long double>(gjk);
    const long double lgkk = static_cast<long double>(gkk);

    const long double lsi = static_cast<long double>(si);
    const long double lsj = static_cast<long double>(sj);
    const long double lsk = static_cast<long double>(sk);

    /*
     * Edge i-j:
     *
     * prediction = t * pi + (1 - t) * pj
     *
     * The unconstrained one-dimensional optimum is then accepted only when
     * it lies strictly inside the edge. Endpoints were already checked above.
     */
    const long double denom_ij = lgii - 2.0L * lgij + lgjj;

    if (denom_ij > 0.0L) {
        const long double numerator_ij = lsi - lsj - lgij + lgjj;

        const long double t = numerator_ij / denom_ij;

        if (
            std::isfinite(t) &&
            t > 0.0L &&
            t < 1.0L
        ) {
            const double wi = static_cast<double>(t);
            const double wj = 1.0 - wi;

            if (wi > 0.0 && wi < 1.0 && wj > 0.0) {
                consider(wi, wj, 0.0);
            }
        }
    }

    /*
     * Edge i-k:
     *
     * prediction = t * pi + (1 - t) * pk
     */
    const long double denom_ik = lgii - 2.0L * lgik + lgkk;

    const long double numerator_ik = lsi - lsk - lgik + lgkk;

    if (denom_ik > 0.0L) {
        const long double t = numerator_ik / denom_ik;

        if (
            std::isfinite(t) &&
            t > 0.0L &&
            t < 1.0L
        ) {
            const double wi = static_cast<double>(t);
            const double wk = 1.0 - wi;

            if (wi > 0.0 && wi < 1.0 && wk > 0.0) {
                consider(wi, 0.0, wk);
            }
        }
    }

    /*
     * Edge j-k:
     *
     * prediction = t * pj + (1 - t) * pk
     */
    const long double denom_jk = lgjj - 2.0L * lgjk + lgkk;

    const long double numerator_jk = lsj - lsk - lgjk + lgkk;

    if (denom_jk > 0.0L) {
        const long double t = numerator_jk / denom_jk;

        if (
            std::isfinite(t) &&
            t > 0.0L &&
            t < 1.0L
        ) {
            const double wj = static_cast<double>(t);
            const double wk = 1.0 - wj;

            if (wj > 0.0 && wj < 1.0 && wk > 0.0) {
                consider(0.0, wj, wk);
            }
        }
    }

    /*
     * Interior.
     *
     * Parameterize:
     *
     *     wk = 1 - wi - wj.
     *
     * Write:
     *
     *     p = pk + wi * (pi - pk) + wj * (pj - pk).
     *
     * The stationary point solves:
     *
     *     [A B] [wi] = [r1]
     *     [B C] [wj]   [r2]
     *
     * where:
     *
     *     A = ||pi - pk||^2
     *     B = (pi - pk)^T (pj - pk)
     *     C = ||pj - pk||^2.
     *
     * denom_ik and denom_jk are exactly A and C.
     */
    const long double cross = lgij - lgik - lgjk + lgkk;

    const long double determinant = denom_ik * denom_jk - cross * cross;

    if (determinant > 0.0L) {
        const long double wi_ld =
            (numerator_ik * denom_jk - cross * numerator_jk) / determinant;

        const long double wj_ld =
            (denom_ik * numerator_jk - cross * numerator_ik) / determinant;

        const long double wk_ld = 1.0L - wi_ld - wj_ld;

        if (
            std::isfinite(wi_ld) &&
            std::isfinite(wj_ld) &&
            std::isfinite(wk_ld) &&
            wi_ld > 0.0L &&
            wj_ld > 0.0L &&
            wk_ld > 0.0L
        ) {
            const double wi = static_cast<double>(wi_ld);
            const double wj = static_cast<double>(wj_ld);

            /*
             * Compute wk from the final double-precision wi and wj so that
             * the returned weights remain as close as possible to a
             * sum-to-one representation after conversion from long double.
             */
            const double wk = 1.0 - wi - wj;

            if (
                std::isfinite(wi) &&
                std::isfinite(wj) &&
                std::isfinite(wk) &&
                wi > 0.0 &&
                wj > 0.0 &&
                wk > 0.0
            ) {
                consider(wi, wj, wk);
            }
        }
    }

    return best;
}


inline void retain_top_candidate(
    std::vector<TripletCandidate>& heap,
    const TripletCandidate& candidate,
    int top_n
) {
    const TripletCandidateLess less;

    if (static_cast<int>(heap.size()) < top_n) {
        heap.push_back(candidate);
        std::push_heap(heap.begin(), heap.end(), less);
        return;
    }

    /*
     * heap.front() is the worst retained candidate. Replace it only when the
     * new candidate is better under the same deterministic total ordering.
     */
    if (less(candidate, heap.front())) {
        std::pop_heap(heap.begin(), heap.end(), less);
        heap.back() = candidate;
        std::push_heap(heap.begin(), heap.end(), less);
    }
}


std::tuple<
    std::vector<
        std::tuple<
            int,
            int,
            int,
            double,
            double,
            double,
            double
        >
    >,
    py::array_t<double>
> find_top_triplets(
    InputArray P_in,
    InputArray y_in,
    int top_n,
    int n_threads = 1
) {
    if (P_in.ndim() != 2) {
        throw py::value_error("P_in must be 2D.");
    }

    if (y_in.ndim() != 1) {
        throw py::value_error("y_in must be 1D.");
    }

    if (P_in.shape(0) != y_in.shape(0)) {
        throw py::value_error(
            "P_in and y_in must have the same number of samples."
        );
    }

    if (P_in.shape(1) < 3) {
        throw py::value_error(
            "Triplet search requires at least 3 model columns."
        );
    }

    if (top_n < 1) {
        throw py::value_error("top_n must be a positive integer.");
    }

    if (n_threads < 1) {
        throw py::value_error("n_threads must be a positive integer.");
    }

    if (
        P_in.shape(0) >
            static_cast<py::ssize_t>(
                std::numeric_limits<int>::max()
            ) ||
        P_in.shape(1) >
            static_cast<py::ssize_t>(
                std::numeric_limits<int>::max()
            )
    ) {
        throw py::value_error(
            "P_in is too large for the native triplet search."
        );
    }

    auto P = P_in.unchecked<2>();
    auto y = y_in.unchecked<1>();

    const int n_samples = static_cast<int>(P.shape(0));

    const int n_models = static_cast<int>(P.shape(1));

    bool saw_zero = false;
    bool saw_one = false;

    for (int row = 0; row < n_samples; ++row) {
        const double y_value = y(row);

        if (!std::isfinite(y_value)) {
            throw py::value_error("y_in must contain only finite values.");
        }

        if (y_value == 0.0) {
            saw_zero = true;
        } else if (y_value == 1.0) {
            saw_one = true;
        } else {
            throw py::value_error(
                "y_in must contain only the binary classes 0 and 1."
            );
        }

        for (int column = 0; column < n_models; ++column) {
            const double probability = P(row, column);

            if (!std::isfinite(probability)) {
                throw py::value_error(
                    "P_in must contain only finite probabilities."
                );
            }

            if (probability < 0.0 || probability > 1.0) {
                throw py::value_error(
                    "P_in probabilities must be between 0 and 1."
                );
            }
        }
    }

    if (!saw_zero || !saw_one) {
        throw py::value_error(
            "y_in must contain exactly the binary classes 0 and 1."
        );
    }

    const std::int64_t model_count = static_cast<std::int64_t>(n_models);

    /*
     * One only needs min(top_n, C(n_models, 3)), not the potentially enormous
     * exact triplet count. If n_models exceeds 2345, C(n_models, 3) is already
     * larger than INT_MAX and therefore larger than any valid positive int
     * top_n. For smaller model counts, the exact calculation is safely within
     * int64_t range.
     */
    int effective_top_n = top_n;

    if (n_models <= 2345) {
        const std::int64_t total_triplets =
            model_count * (model_count - 1) * (model_count - 2) / 6;

        effective_top_n =
            static_cast<int>(
                std::min<std::int64_t>(
                    static_cast<std::int64_t>(top_n),
                    total_triplets
                )
            );
    }

    OmpThreadGuard thread_guard(n_threads);

    /*
     * Precompute:
     *
     *     G  = P^T P
     *     s  = P^T y
     *     y2 = y^T y
     *
     * Then every triplet's Brier SSE can be evaluated using only the relevant
     * entries of G and s, avoiding an O(n_samples) loop for every triplet.
     */
    Eigen::MatrixXd G(n_models, n_models);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_models; ++i) {
        for (int j = i; j < n_models; ++j) {
            double sum = 0.0;

            for (int row = 0; row < n_samples; ++row) {
                sum += P(row, i) * P(row, j);
            }

            G(i, j) = sum;
            G(j, i) = sum;
        }
    }

    Eigen::VectorXd s(n_models);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_models; ++i) {
        double sum = 0.0;

        for (int row = 0; row < n_samples; ++row) {
            sum += P(row, i) * y(row);
        }

        s(i) = sum;
    }

    /*
     * Keep this accumulation serial. The target is small relative to the
     * exhaustive triplet search, and serial accumulation avoids any
     * reduction-order dependence.
     */
    double y2 = 0.0;

    for (int row = 0; row < n_samples; ++row) {
        y2 += y(row) * y(row);
    }

    std::vector<TripletCandidate> global_best;
    global_best.reserve(
        static_cast<std::size_t>(effective_top_n)
    );

    /*
     * Exhaustively enumerate every unique i < j < k triplet.
     *
     * Each OpenMP worker maintains only its own top-N candidates. Those local
     * top-N sets are merged into one global top-N heap after the worker's
     * assigned iterations are complete.
     */
    #pragma omp parallel
    {
        std::vector<TripletCandidate> local_best;

        local_best.reserve(static_cast<std::size_t>(effective_top_n));

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < n_models - 2; ++i) {
            const double gii = G(i, i);
            const double si = s(i);

            for (int j = i + 1; j < n_models - 1; ++j) {
                const double gij = G(i, j);
                const double gjj = G(j, j);
                const double sj = s(j);

                for (int k = j + 1; k < n_models; ++k) {
                    const double gik = G(i, k);
                    const double gjk = G(j, k);
                    const double gkk = G(k, k);
                    const double sk = s(k);

                    const TripletSolution solution =
                        solve_convex_triplet(
                            gii,
                            gij,
                            gik,
                            gjj,
                            gjk,
                            gkk,
                            si,
                            sj,
                            sk,
                            y2
                        );

                    const double brier_score =
                        static_cast<double>(
                            solution.sse / static_cast<long double>(n_samples)
                        );

                    if (!std::isfinite(brier_score)) {
                        continue;
                    }

                    const TripletCandidate candidate{
                        brier_score,
                        i,
                        j,
                        k,
                        solution.wi,
                        solution.wj,
                        solution.wk
                    };

                    retain_top_candidate(
                        local_best,
                        candidate,
                        effective_top_n
                    );
                }
            }
        }

        #pragma omp critical(triplet_top_n_merge)
        {
            for (const TripletCandidate& candidate : local_best) {
                retain_top_candidate(
                    global_best,
                    candidate,
                    effective_top_n
                );
            }
        }
    }

    if (global_best.empty()) {
        throw std::runtime_error(
            "Convex triplet search did not produce any candidates."
        );
    }

    std::sort(
        global_best.begin(),
        global_best.end(),
        TripletCandidateLess{}
    );

    const int n_kept = static_cast<int>(global_best.size());

    std::vector<
        std::tuple<
            int,
            int,
            int,
            double,
            double,
            double,
            double
        >
    > metadata;

    metadata.reserve(static_cast<std::size_t>(n_kept));

    py::array_t<double> probabilities_out(
        {
            static_cast<py::ssize_t>(n_samples),
            static_cast<py::ssize_t>(n_kept)
        }
    );

    auto probabilities = probabilities_out.mutable_unchecked<2>();

    for (int column = 0; column < n_kept; ++column) {
        const TripletCandidate& candidate = global_best[column];

        metadata.emplace_back(
            candidate.i,
            candidate.j,
            candidate.k,
            candidate.wi,
            candidate.wj,
            candidate.wk,
            candidate.brier_score
        );

        for (int row = 0; row < n_samples; ++row) {
            probabilities(row, column) =
                candidate.wi * P(row, candidate.i) +
                candidate.wj * P(row, candidate.j) +
                candidate.wk * P(row, candidate.k);
        }
    }

    return std::make_tuple(
        metadata,
        probabilities_out
    );
}


}  // namespace


PYBIND11_MODULE(_triplet_search, m) {
    m.doc() =
        "Exact exhaustive Brier-optimal convex triplet ensemble search "
        "with OpenMP and Eigen";

    m.def(
        "find_top_triplets",
        &find_top_triplets,
        py::arg("P_in"),
        py::arg("y_in"),
        py::arg("top_n"),
        py::arg("n_threads") = 1
    );
}
