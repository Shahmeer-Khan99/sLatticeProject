#ifndef POINTER2D_H
#define POINTER2D_H 

#include<iostream>
#include<stdio.h>

struct Pointer2D {
  int** arr;
  int* sizes;
  int* capacity;
  int rows;

  Pointer2D(int r , int initialCapacity = 4) {
    rows = r;
    arr = new int*[r];
    sizes = new int[r]();
    capacity = new int[r];
    for(int i = 0; i < rows; i++){
      arr[i] = new int[initialCapacity];
      capacity[i] = initialCapacity;
    }    
  }

  int* getRow(int row) {
    return arr[row];
  }

  int getRowSize(int row) {
    return sizes[row];
  }

  void push(int row, int val) {
    if(sizes[row] == capacity[row]) {
      int newCap = capacity[row] * 2;
      int* newRow = new int[newCap];
      for(int i = 0; i < sizes[row]; i++) {
        newRow[i] = arr[row][i];
      }
      delete[] arr[row];
      arr[row] = newRow;
      capacity[row] = newCap;
    }
    arr[row][sizes[row]++] = val;
  }

  void shift(int row) {
    if(sizes[row] <= 0) return;
    for(int i = 1; i < sizes[row]; i++) {
      arr[row][i - 1] = arr[row][i];
    }

    sizes[row]--;
  }

  bool empty(int row) {
    if(sizes[row] <= 0) return true;
    return false;
  }
};

#endif