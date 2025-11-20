#include <iostream>
#include<stdio.h>
#include<vector>
#include<fstream>
#include <limits>
#include <string>
#include <variant>
#include <sstream>
#include <algorithm>
#include <cuda.h>
#include <chrono>
#include <unordered_map>


struct Variable {
  int index;
  int lb;
  int ub;
};

struct Constraint {
  int left;
  int right;
  int resultant;
  char op; // '+', '-', '*', '/', 'A', 'I', 'L'
  char deprecated;
};

struct UnionFind {
  int n;
  std::vector<int> parent;
  std::vector<int> rank;
  UnionFind(int n) : n(n), parent(n), rank(n, 0) {
      for (int i = 0; i < n; ++i) parent[i] = i;
  }
  int find(int x) {
      if (parent[x] != x) parent[x] = find(parent[x]);
      return parent[x];
  }
  void unite(int a, int b) {
      a = find(a);
      b = find(b);
      if (a == b) return;
      if (rank[a] < rank[b]) parent[a] = b;
      else if (rank[a] > rank[b]) parent[b] = a;
      else { parent[b] = a; rank[a]++; }
  }
};

struct Equivalence {
  UnionFind uf;
  Equivalence(int rows): uf(rows) {}
};

__device__ int find_gpu(int* parent, int x) {
  while(parent[x] != x) {
      parent[x] = parent[parent[x]]; // path compression
      x = parent[x];
  }
  return x;
}

__device__ void unite_gpu(int* parent, int* rank, int a, int b) {
  a = find_gpu(parent, a);
  b = find_gpu(parent, b);
  if(a == b) return;

  if(rank[a] < rank[b]) parent[a] = b;
  else if(rank[a] > rank[b]) parent[b] = a;
  else { parent[b] = a; rank[a]++; }
}

// // ---------------- GPU replaceVars ----------------
// __global__ void gpu_replaceVars(
//   Variable* vars,
//   Constraint* constraints,
//   int* parent,
//   int* rank,
//   int* constraintFlags,
//   int constraintLength
// ) {
//   int idx = threadIdx.x + blockDim.x * blockIdx.x;
//   if(idx >= constraintLength) return;

//   if(constraintFlags[idx] == 1) {
//       int left = constraints[idx].left;
//       int right = constraints[idx].right;

//       // unite left/right in GPU union-find
//       unite_gpu(parent, rank, left, right);

//       // representative
//       int rep = find_gpu(parent, left);

//       // update variables to new representative
//       vars[left].index = rep;
//       vars[right].index = rep;

//       // update constraint itself
//       if(constraints[idx].left == left || constraints[idx].left == right)
//           constraints[idx].left = rep;
//       if(constraints[idx].right == left || constraints[idx].right == right)
//           constraints[idx].right = rep;
//       if(constraints[idx].resultant == left || constraints[idx].resultant == right)
//           constraints[idx].resultant = rep;
//   }
// }

// __device__ bool merge(Variable* vars, Constraint* constraints, int index) {
//   int left = constraints[index].left;
//   int right = constraints[index].right;
//   int resultant = constraints[index].resultant;

//   if(vars[resultant].lb == 1 && vars[resultant].ub == 1) {
//     int mergedUB = vars[left].ub < vars[right].ub ? vars[left].ub : vars[right].ub ;
//     int mergedLB = vars[left].lb > vars[right].lb ? vars[left].lb : vars[right].lb ;

//     if (mergedLB > mergedUB) {
//       mergedLB = mergedUB;
//     }
    
//     vars[constraints[index].left].lb = mergedLB;
//     vars[constraints[index].right].lb = mergedLB;
//     vars[constraints[index].left].ub = mergedUB;
//     vars[constraints[index].right].ub= mergedUB;

//     return true;
//   }

//   return false;
// }

// __global__ void gpu_launch(Variable* vars, Constraint* constraints, bool *changed, int* constraintFlags, int constraintLength) {
//   int idx = threadIdx.x + (blockDim.x * blockIdx.x);
//   if(idx >= constraintLength) return;
//   bool deprecate;
//   if(constraints[idx].op == '=' && constraints[idx].deprecated != 'R') {
//     deprecate = merge(vars, constraints, idx);
//     if(deprecate) {
//       *changed = false;
//       constraints[idx].deprecated = 'R';
//       constraintFlags[idx] = 1;
//     }
//   }
// }


__device__ int find_gpu_atomic(int* parent, int x) {
  int p = parent[x];
  while (p != parent[p]) {
      int pp = parent[p];
      atomicCAS(&parent[x], p, pp);
      p = parent[x];
  }
  return p;
}

__device__ void unite_gpu_atomic(int* parent, int* rank, int a, int b) {
  while (true) {
    a = find_gpu_atomic(parent, a);
    b = find_gpu_atomic(parent, b);
    if (a == b) return;

    int ra = rank[a];
    int rb = rank[b];
    if (ra < rb) {
        if (atomicCAS(&parent[a], a, b) == a) return;
    } else if (ra > rb) {
        if (atomicCAS(&parent[b], b, a) == b) return;
    } else {
        if (atomicCAS(&parent[b], b, a) == b) {
            atomicAdd(&rank[a], 1);
            return;
        }
    }
  }
}

__global__ void gpu_merge_constraints(
  Variable* vars,
  Constraint* constraints,
  int* parent,
  int* rank,
  int* changed,
  int constraintLength
) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= constraintLength) return;

  Constraint &c = constraints[idx];
  if (c.deprecated == 'R') return;

  int x = c.resultant;
  int y = c.left;
  int z = c.right;

  if (c.op == '=') {
      if (vars[x].lb == 1 && vars[x].ub == 1) {

          int mergedLB = max(vars[y].lb, vars[z].lb);
          int mergedUB = min(vars[y].ub, vars[z].lb);
          if (mergedLB > mergedUB) mergedLB = mergedUB;

          atomicMax(&vars[y].lb, mergedLB);
          atomicMax(&vars[z].lb, mergedLB);
          atomicMin(&vars[y].ub, mergedUB);
          atomicMin(&vars[z].ub, mergedUB);

          unite_gpu_atomic(parent, rank, y, z);

          int rep = find_gpu_atomic(parent, y);
          c.left = rep;
          c.right = rep;
          c.resultant = rep;

          c.deprecated = 'R';
          atomicAnd(changed, 0);
      }
      return;
  }

  if (c.op == '+') {
      if (vars[z].lb == 0 && vars[z].ub == 0) {
          unite_gpu_atomic(parent, rank, x, y);
          c.deprecated = 'R';
          atomicAnd(changed, 0);
          return;
      }

      if (vars[y].lb == 0 && vars[y].ub == 0) {
          unite_gpu_atomic(parent, rank, x, z);
          c.deprecated = 'R';
          atomicAnd(changed, 0);
          return;
      }

      return;
  }

  if (c.op == '*') {
      if (vars[z].lb == 1 && vars[z].ub == 1) {
          unite_gpu_atomic(parent, rank, x, y);
          c.deprecated = 'R';
          atomicAnd(changed, 0);
          return;
      }

      if (vars[y].lb == 1 && vars[y].ub == 1) {
          unite_gpu_atomic(parent, rank, x, z);
          c.deprecated = 'R';
          atomicAnd(changed, 0);
          return;
      }
      return;
  }
}

void simplify(std::vector<Variable>& vars, std::vector<Constraint>& constraints, Equivalence& equivalences, size_t varLength, size_t constraintLength) {
  Variable* d_vars;
  Constraint* d_constraints;

  cudaMalloc(&d_vars, sizeof(Variable) * varLength);
  cudaMalloc(&d_constraints, sizeof(Constraint) * constraintLength);

  int* d_parent;
  int* d_rank;
  cudaMalloc(&d_parent, sizeof(int) * varLength);
  cudaMalloc(&d_rank, sizeof(int) * varLength);

  std::vector<int> parent(varLength), rank(varLength, 0);
  for (int i = 0; i < varLength; ++i) parent[i] = i;

  cudaMemcpy(d_parent, parent.data(), sizeof(int) * varLength, cudaMemcpyHostToDevice);
  cudaMemcpy(d_rank, rank.data(), sizeof(int) * varLength, cudaMemcpyHostToDevice);

  cudaMemcpy(d_vars, vars.data(), sizeof(Variable) * varLength, cudaMemcpyHostToDevice);
  cudaMemcpy(d_constraints, constraints.data(), sizeof(Constraint) * constraintLength, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (constraintLength + threads - 1) / threads;

  int h_changed = 1;
  int* d_changed;
  cudaMalloc(&d_changed, sizeof(int));

  while (h_changed) {
      h_changed = 1;
      cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);

      gpu_merge_constraints<<<blocks, threads>>>(
          d_vars, d_constraints, d_parent, d_rank, d_changed, constraintLength
      );
      cudaDeviceSynchronize();

      cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
  }

  cudaMemcpy(vars.data(), d_vars, sizeof(Variable) * varLength, cudaMemcpyDeviceToHost);
  cudaMemcpy(constraints.data(), d_constraints, sizeof(Constraint) * constraintLength, cudaMemcpyDeviceToHost);
  cudaMemcpy(parent.data(), d_parent, sizeof(int) * varLength, cudaMemcpyDeviceToHost);

  for (int i = 0; i < varLength; ++i)
      equivalences.uf.parent[i] = parent[i];

  for (int i = 0; i < varLength; ++i) {
      int rep = equivalences.uf.find(i);
      vars[i].index = rep;
  }

  cudaFree(d_vars);
  cudaFree(d_constraints);
  cudaFree(d_parent);
  cudaFree(d_rank);
  cudaFree(d_changed);
}


void print(std::vector<Variable>& vars, std::vector<Constraint>& constraints, Equivalence& equivalences) {
  std::cout << "VARIABLES" << std::endl;
  for(int i = 0; i < vars.size(); i++) {
    std::cout << "Variable: " <<  i <<  std::endl;
    if(vars[i].index == i) {
      std::cout << "Index: " <<  vars[i].index <<  std::endl;
      std::cout << "Lower Bound: " << vars[i].lb << std::endl;
      std::cout << "Upper Bound: " << vars[i].ub << std::endl; 
    } else {
      std::cout << "Repsentative Variable: " <<  vars[i].index <<  std::endl;
    }
  }

  std::cout << "CONSTRAINTS" << std::endl;
  for(int j = 0; j < constraints.size(); j++) {
    std::cout << "Constraint: " << j << std::endl;
    std::cout <<  constraints[j].deprecated << " " << constraints[j].resultant <<  " = " << constraints[j].left << " " << constraints[j].op << " " << constraints[j].right << std::endl;
  }

  std::cout << "\nEQUIVALENCE CLASSES" << std::endl;

  std::unordered_map<int, std::vector<int>> groups;
  for (int i = 0; i < (int)vars.size(); ++i) {
      int rep = equivalences.uf.find(i);
      groups[rep].push_back(i);
  }
  for (auto &kv : groups) {
      std::cout << "{ ";
      for (int v : kv.second) std::cout << v << " ";
      std::cout << "}" << std::endl;
  }

}

void writeOutput(const std::vector<Variable>& vars, const std::vector<Constraint>& constraints, Equivalence& equivalences, const std::string& filename) {
  std::ofstream file(filename);
  if (!file) {
    std::cerr << "Error: Could not open file " << filename << " for writing!" << std::endl;
    return;
  }

  file << vars.size() << std::endl;
  for (const auto& v : vars) {
    if (v.index == &v - &vars[0]) { // representative
      file << v.lb << " " << v.ub << std::endl;
    } else { // non-representative
      file << v.index << std::endl;
    }
  }

  file << constraints.size() << std::endl;
  for (const auto& c : constraints) {
    if (c.deprecated == 'R') file << "R ";
    file << c.resultant << " " << c.left << " " << c.op << " " << c.right << std::endl;
  }

  std::unordered_map<int, std::vector<int>> groups;
  for (int i = 0; i < vars.size(); ++i) {
    int rep = equivalences.uf.find(i);
    groups[rep].push_back(i);
  }

  file << groups.size() << std::endl;
  for (auto& kv : groups) {
  file << "{ ";
  for (int v : kv.second) file << v << " ";
    file << "}" << std::endl;
  }

  file.close();
}


int main(int argc, char* argv[]) {
  std::string input_file = "data.tcn";
  std::string output_file;

  for (int i = 1; i < argc; i++) {
      if (std::string(argv[i]) == "-o" && i + 1 < argc) {
          output_file = argv[++i];
      } else {
          input_file = argv[i];
      }
  }

  using Token  = std::variant<int, char>;
  using Line   = std::vector<Token>;
  using Block  = std::vector<Line>;

  std::ifstream file(input_file);
  if (!file) {
      std::cerr << "Error: Could not open file " << input_file << "!" << std::endl;
      return 1;
  }

  std::streampos start = file.tellg();
  int numVariables; 
  file >> numVariables;
  file.clear();
  file.seekg(start);

  Equivalence equivalenceVars(numVariables);

  int n;
  std::vector<Block> blocks;

  auto is_integer = [](const std::string& s) {
      if (s.empty()) return false;
      size_t start = (s[0] == '-' ? 1 : 0);
      if (start == s.size()) return false; // "-" alone is invalid
      return std::all_of(s.begin() + start, s.end(), ::isdigit);
  };

  while (file >> n) {
      file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

      Block block;
      std::string line;

      for (int i = 0; i < n && std::getline(file, line); ++i) {
          if (line.empty()) continue;

          std::istringstream iss(line);
          Line tokens;

          std::string tok;
          while (iss >> tok) {
              if (is_integer(tok)) {
                  tokens.push_back(std::stoi(tok));
              } else {
                  tokens.push_back(tok[0]); // operator
              }
          }

          block.push_back(tokens);
      }

      blocks.push_back(block);
  }

  file.close();

  std::vector<Variable> vars;
  std::vector<Constraint> constraints;

  for (int i = 0; i < blocks.size(); i++) {
      if (i == 0) {
          Block varBlock = blocks[0];
          for (int j = 0; j < varBlock.size(); j++) {
              Line varData = varBlock[j];
              Variable var;
              var.index = j;
              var.lb = std::get<int>(varData[0]);
              var.ub = std::get<int>(varData[1]);
              vars.push_back(var);
          }
      } 
      else {
          Block constraintBlock = blocks[1];
          for (int k = 0; k < constraintBlock.size(); k++) {
              Line cData = constraintBlock[k];
              Constraint c;
              c.resultant = std::get<int>(cData[0]);
              c.left      = std::get<int>(cData[1]);
              c.op        = std::get<char>(cData[2]);
              c.right     = std::get<int>(cData[3]);
              c.deprecated = 0;
              constraints.push_back(c);
          }
      }
  }

  auto t_start = std::chrono::high_resolution_clock::now();
  simplify(vars, constraints, equivalenceVars, blocks[0].size(), blocks[1].size());
  cudaDeviceSynchronize();
  auto t_end = std::chrono::high_resolution_clock::now();

  double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  
  std::cout << "simplify() took " << ms << " ms\n";

  if (!output_file.empty()) {
      writeOutput(vars, constraints, equivalenceVars, output_file);
      std::cout << "Simplified TCN written to: " << output_file << std::endl;
  } else {
      print(vars, constraints, equivalenceVars);
  }

  return 0;
}

