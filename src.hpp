#pragma once
#include "simulator.hpp"

namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  Matrix *accum_k = nullptr;
  Matrix *accum_v = nullptr;

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    size_t M = i + 1;
    size_t N = i + 1;
    std::vector<Matrix *> temps;

    auto Alloc = [&](const std::string &name = "") -> Matrix * {
      Matrix *m = matrix_memory_allocator.Allocate(name);
      temps.push_back(m);
      return m;
    };

    if (i == 0) {
      accum_k = keys[0];
      accum_v = values[0];
    } else {
      Matrix *new_k = matrix_memory_allocator.Allocate();
      Matrix *new_v = matrix_memory_allocator.Allocate();
      gpu_sim.Concat(accum_k, keys[i], new_k, 0, Position::kInGpuHbm);
      gpu_sim.Concat(accum_v, values[i], new_v, 0, Position::kInGpuHbm);
      if (i > 1) {
        gpu_sim.ReleaseMatrix(accum_k);
        gpu_sim.ReleaseMatrix(accum_v);
      }
      accum_k = new_k;
      accum_v = new_v;
    }

    Matrix *accum_o = nullptr;
    const size_t CHUNK = 4;
    for (size_t r_start = 0; r_start < M; r_start += CHUNK) {
      size_t r_end = std::min(M, r_start + CHUNK);
      size_t cur_M = r_end - r_start;

      // Extract Q chunk
      Matrix *q_chunk_hbm = nullptr;
      for (size_t r = r_start; r < r_end; ++r) {
        Matrix *row_r = Alloc();
        gpu_sim.GetRow(current_query, r, row_r, Position::kInGpuHbm);
        if (r == r_start) {
          q_chunk_hbm = row_r;
        } else {
          Matrix *new_q_chunk = Alloc();
          gpu_sim.Concat(q_chunk_hbm, row_r, new_q_chunk, 0, Position::kInGpuHbm);
          gpu_sim.ReleaseMatrix(q_chunk_hbm);
          gpu_sim.ReleaseMatrix(row_r);
          q_chunk_hbm = new_q_chunk;
        }
      }

      // Compute A_chunk = Q_chunk * K^T column by column
      Matrix *accum_a = nullptr;
      for (size_t d = 0; d < 512; ++d) {
        Matrix *q_col = Alloc();
        gpu_sim.GetColumn(q_chunk_hbm, d, q_col, Position::kInGpuHbm);
        gpu_sim.MoveMatrixToSharedMem(q_col);

        Matrix *k_col = Alloc();
        gpu_sim.GetColumn(accum_k, d, k_col, Position::kInGpuHbm);
        gpu_sim.MoveMatrixToSharedMem(k_col);
        gpu_sim.Transpose(k_col, Position::kInSharedMemory);

        Matrix *prod_d = Alloc();
        gpu_sim.MatMul(q_col, k_col, prod_d);
        gpu_sim.ReleaseMatrix(q_col);
        gpu_sim.ReleaseMatrix(k_col);

        if (d == 0) {
          accum_a = prod_d;
        } else {
          Matrix *new_accum_a = Alloc();
          gpu_sim.MatAdd(accum_a, prod_d, new_accum_a);
          gpu_sim.ReleaseMatrix(accum_a);
          gpu_sim.ReleaseMatrix(prod_d);
          accum_a = new_accum_a;
        }
      }
      gpu_sim.ReleaseMatrix(q_chunk_hbm);

      // Compute Softmax(A_chunk) row by row
      Matrix *softmax_a = nullptr;
      for (size_t r = 0; r < cur_M; ++r) {
        Matrix *row_r = Alloc();
        gpu_sim.GetRow(accum_a, r, row_r, Position::kInSharedMemory);

        Matrix *exp_r = Alloc();
        gpu_sim.MatExp(row_r, exp_r);
        gpu_sim.ReleaseMatrix(row_r);

        Matrix *sum_r = Alloc();
        gpu_sim.Sum(exp_r, sum_r);

        Matrix *soft_row = Alloc();
        gpu_sim.MatDiv(exp_r, sum_r, soft_row);
        gpu_sim.ReleaseMatrix(exp_r);
        gpu_sim.ReleaseMatrix(sum_r);

        if (r == 0) {
          softmax_a = soft_row;
        } else {
          Matrix *new_softmax_a = Alloc();
          gpu_sim.Concat(softmax_a, soft_row, new_softmax_a, 0, Position::kInSharedMemory);
          gpu_sim.ReleaseMatrix(softmax_a);
          gpu_sim.ReleaseMatrix(soft_row);
          softmax_a = new_softmax_a;
        }
      }
      gpu_sim.ReleaseMatrix(accum_a);

      // Compute Out_chunk = Softmax(A_chunk) * V column by column
      Matrix *accum_o_chunk = nullptr;
      for (size_t d = 0; d < 512; ++d) {
        Matrix *v_col = Alloc();
        gpu_sim.GetColumn(accum_v, d, v_col, Position::kInGpuHbm);
        gpu_sim.MoveMatrixToSharedMem(v_col);

        Matrix *o_col = Alloc();
        gpu_sim.MatMul(softmax_a, v_col, o_col);
        gpu_sim.ReleaseMatrix(v_col);

        gpu_sim.MoveMatrixToGpuHbm(o_col);

        if (d == 0) {
          accum_o_chunk = o_col;
        } else {
          Matrix *new_accum_o_chunk = Alloc();
          gpu_sim.Concat(accum_o_chunk, o_col, new_accum_o_chunk, 1, Position::kInGpuHbm);
          gpu_sim.ReleaseMatrix(accum_o_chunk);
          gpu_sim.ReleaseMatrix(o_col);
          accum_o_chunk = new_accum_o_chunk;
        }
      }
      gpu_sim.ReleaseMatrix(softmax_a);

      // Accumulate into accum_o in HBM
      if (r_start == 0) {
        accum_o = accum_o_chunk;
      } else {
        Matrix *new_accum_o = Alloc();
        gpu_sim.Concat(accum_o, accum_o_chunk, new_accum_o, 0, Position::kInGpuHbm);
        gpu_sim.ReleaseMatrix(accum_o);
        gpu_sim.ReleaseMatrix(accum_o_chunk);
        accum_o = new_accum_o;
      }
    }

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*accum_o);

    for (auto *m : temps) {
      delete m;
    }
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
