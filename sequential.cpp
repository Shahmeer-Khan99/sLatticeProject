#include <iostream>
#include<stdio.h>
#include<vector>
#include<fstream>
#include <limits>
#include <string>
#include <variant>
#include <sstream>
#include <algorithm>
#include "doublePointer.h"


struct Variable {
  int index;
  int lb;
  int ub;
};

struct Constraint {
  int left;
  int right;
  int resultant;
  std::string op; // '+', '-', '*', '/', 'A', 'I', 'L'
  std::string deprecated;
};

struct Equivalence {
  Pointer2D variables;
  Equivalence(int rows): variables(rows) {}
};

void constraintRedesign(std::vector<Variable>& vars, std::vector<Constraint>& constraints, int* changed, size_t changed_size) {
  int representative = changed[0];
  for(int i = 1; i < changed_size; i++) {
    for(int j = 0; j < constraints.size(); j++) {
      if(constraints[j].left == changed[i]) {
        constraints[j].left = representative;
      } else if(constraints[j].right == changed[i]) {
        constraints[j].right = representative;
      } else if(constraints[j].resultant == changed[i]) {
        constraints[j].resultant = representative;
      }
    }
  }
}

bool mergeDomains(std::vector<Variable>& vars, std::vector<Constraint>& constraints, int index) {
  int left = constraints[index].left;
  int right = constraints[index].right;
  int resultant = constraints[index].resultant;

  if(vars[resultant].lb == 1 && vars[resultant].ub == 1) {
    int mergedUB = std::min(vars[left].ub, vars[right].ub);
    int mergedLB = std::max(vars[left].lb, vars[right].lb);

    if (mergedLB > mergedUB) {
      std::cerr << "Warning: Inconsistent domain between " 
                << left << " and " << right << " (no overlap)\n";
      mergedLB = mergedUB;
    }
  
    std::vector<int> looper = { constraints[index].left,  constraints[index].right };
  
    for(auto& var: looper) {
      vars[var].lb = mergedLB;
      vars[var].ub = mergedUB;
    }

    return true;
  }

  return false;
}

void replaceVars(std::vector<Variable>& vars, std::vector<Constraint>& constraints, int index, Equivalence& equivalences) {
  int left = constraints[index].left;
  int right = constraints[index].right;
  int resultant = constraints[index].resultant;
  Pointer2D equivalencesArray = equivalences.variables;
  // std::vector<int> changed;
  int changed[3];

  if(vars[resultant].lb == 1 && vars[resultant].ub == 1) {
    int representativeIndex = std::min(left, right);
    if(representativeIndex == left) {
      equivalencesArray.push(representativeIndex, right);
      equivalencesArray.shift(right);
      vars[right].index = representativeIndex;
    } else {
      equivalencesArray.push(representativeIndex, left);
      equivalencesArray.shift(left);
      vars[left].index = representativeIndex;
    }

    changed[0] = representativeIndex;
    changed[1] = left;
    changed[2] = right;
    constraintRedesign(vars, constraints, changed, 3);
  }
}

void simplify(std::vector<Variable>& vars, std::vector<Constraint>& constraints, Equivalence& equivalences) {
  bool flag = false;
  bool deprecate;
  while(!flag) {
    flag = true;
    for(int i = 0; i < constraints.size(); i++) {
      if(constraints[i].op == "=" && constraints[i].deprecated != "R") {
        deprecate = mergeDomains(vars, constraints, i);
        if(deprecate) {
          constraints[i].deprecated = "R";
          flag = false;
        }
        replaceVars(vars, constraints, i, equivalences);
      }
    }
  }
}

void print(std::vector<Variable>& vars, std::vector<Constraint>& constraints, Equivalence& equivalences) {
  Pointer2D equivalencesArray = equivalences.variables;
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

  std::vector<bool> printed(vars.size(), false);
  for (int i = 0; i < vars.size(); i++) {
      if (!equivalencesArray.empty(i) && vars[i].index == i) {
          std::cout << "{ ";
          for (int j = 0; j < equivalencesArray.getRowSize(i); j++) {
              std::cout << equivalencesArray.getRow(i)[j] << " ";
              printed[equivalencesArray.getRow(i)[j] ] = true;
          }
          std::cout << "}" << std::endl;
      } else if (!printed[i] && vars[i].index == i) {
          // isolated variable (never merged)
          std::cout << "{ " << i << " }" << std::endl;
      }
  }

}

int main() {
  using Token  = std::variant<int, std::string>;
  using Line = std::vector<Token>;
  using Block = std::vector<Line>;

  std::ifstream file("data.tcn");
  if (!file) {
    std::cerr << "Error: Could not open file!" << std::endl;
    return 1;
  }

  std::streampos start = file.tellg();
  int numVariables; file >> numVariables;
  file.clear(); file.seekg(start);

  Equivalence equivalenceVars(numVariables);
  for(int j = 0; j < numVariables; j++) {
    // std::vector<int> vec;
    // vec.push_back(j);
    equivalenceVars.variables.push(j, j);
  }

  int n;
  std::vector<Block> blocks;
  while (file >> n) {
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    Block block;
    std::string line;

    for (int i = 0; i < n && std::getline(file, line); ++i) {
      if (line.empty()) continue;

      std::istringstream iss(line);
      std::string word;
      Line tokens;

      while(iss >> word) {
        try {
          int val = std::stoi(word);
          tokens.push_back(val); 
        } catch (...) {
          tokens.push_back(word);
        }
      }
      block.push_back(tokens);
    }

    blocks.push_back(block);
  }

  file.close();

  std::vector<Variable> vars;
  std::vector<Constraint> constraints;

  for(int i = 0; i < blocks.size(); i++) {
    if(i == 0) {
      Block varBlock = blocks[0];
      for(int j = 0; j < varBlock.size(); j++) {
        Line varData = varBlock[j];
        Variable var;
        var.index = j;
        var.lb = std::get<int>(varData[0]);
        var.ub = std::get<int>(varData[1]);
        vars.push_back(var);
      }
    } else {
      Block constraintBlock = blocks[1];
      for(int k = 0; k < constraintBlock.size(); k++) {
        Line constraintData = constraintBlock[k];
        Constraint constraint;
        constraint.resultant = std::get<int>(constraintData[0]);
        constraint.left = std::get<int>(constraintData[1]);
        constraint.op = std::get<std::string>(constraintData[2]);
        constraint.right = std::get<int>(constraintData[3]);
        constraint.deprecated = ' ';
        constraints.push_back(constraint);
      }
    }
  }

  print(vars, constraints, equivalenceVars);

  simplify(vars, constraints, equivalenceVars);

  std::cout << "AFTERRR" << std::endl;

  print(vars, constraints, equivalenceVars);
  
  return 0;
}
