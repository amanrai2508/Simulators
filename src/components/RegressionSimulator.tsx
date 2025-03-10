"use client"

import React, { useState, useEffect } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Slider } from '@/components/ui/slider';

interface DataPoint {
  x1: number;
  x2: number;
  x3: number;
  y: number;
  id: number;
}

interface Coefficients {
  ols: number[];
  ridge: number[];
  lasso: number[];
}

const RegressionSimulator = () => {
  // State variables
  const [data, setData] = useState<DataPoint[]>([]);
  const [lambdaRidge, setLambdaRidge] = useState([1]);
  const [lambdaLasso, setLambdaLasso] = useState([1]);
  const [coefficients, setCoefficients] = useState<Coefficients>({
    ols: [0, 0, 0, 0], 
    ridge: [0, 0, 0, 0], 
    lasso: [0, 0, 0, 0]
  });
  const [regressionType, setRegressionType] = useState('ridge');

  // Generate random data
  useEffect(() => {
    generateData();
  }, []);

  const generateData = () => {
    const newData: DataPoint[] = [];
    // True coefficients
    const trueCoefs = [2, 1, 0.5, 0.1];
    
    for (let i = 0; i < 50; i++) {
      const x1 = Math.random() * 10 - 5;
      const x2 = Math.random() * 10 - 5;
      const x3 = Math.random() * 10 - 5;
      
      // Generate y with some noise
      const y = trueCoefs[0] + trueCoefs[1] * x1 + trueCoefs[2] * x2 + trueCoefs[3] * x3 + (Math.random() * 3 - 1.5);
      
      newData.push({
        x1, x2, x3, y,
        id: i
      });
    }
    
    setData(newData);
    calculateRegressions(newData, lambdaRidge[0], lambdaLasso[0]);
  };

  // Calculate OLS, Ridge, and Lasso regressions
  const calculateRegressions = (dataPoints: DataPoint[], ridgeLambda: number, lassoLambda: number) => {
    // Simple implementation of matrix operations for linear regression
    
    // Create X matrix (features) and Y vector (target)
    const X = dataPoints.map(point => [1, point.x1, point.x2, point.x3]);
    const Y = dataPoints.map(point => point.y);
    
    // Calculate OLS coefficients: β = (X^T X)^(-1) X^T Y
    const XtX = matrixMultiply(transpose(X), X);
    const XtY = matrixMultiply(transpose(X), Y.map(y => [y])).map(row => row[0]);
    const olsCoef = solveLinearSystem(XtX, XtY);
    
    // Calculate Ridge coefficients: β = (X^T X + λI)^(-1) X^T Y
    const ridgeXtX = XtX.map((row, i) => 
      row.map((val, j) => i === j ? val + ridgeLambda : val)
    );
    const ridgeCoef = solveLinearSystem(ridgeXtX, XtY);
    
    // For Lasso, use a simple coordinate descent algorithm (simplified)
    // Start with OLS solution and shrink coefficients
    const lassoCoef = [...olsCoef];
    for (let iter = 0; iter < 100; iter++) {
      for (let j = 0; j < lassoCoef.length; j++) {
        if (j === 0) continue; // Skip intercept
        
        // Soft thresholding operator
        const sign = lassoCoef[j] > 0 ? 1 : (lassoCoef[j] < 0 ? -1 : 0);
        const magnitude = Math.abs(lassoCoef[j]);
        
        lassoCoef[j] = sign * Math.max(0, magnitude - lassoLambda / 20);
      }
    }
    
    setCoefficients({
      ols: olsCoef,
      ridge: ridgeCoef,
      lasso: lassoCoef
    });
  };
  
  // Helper function: Matrix transpose
  const transpose = (matrix: number[][]) => {
    const rows = matrix.length;
    const cols = matrix[0].length;
    const result = Array(cols).fill(0).map(() => Array(rows).fill(0));
    
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result[j][i] = matrix[i][j];
      }
    }
    
    return result;
  };
  
  // Helper function: Matrix multiplication
  const matrixMultiply = (A: number[][], B: number[][] | number[]) => {
    const rowsA = A.length;
    const colsA = A[0].length;
    const isMatrix = Array.isArray(B[0]);
    const colsB = isMatrix ? (B[0] as number[]).length : 1;
    const result = Array(rowsA).fill(0).map(() => Array(colsB).fill(0));
    
    for (let i = 0; i < rowsA; i++) {
      for (let j = 0; j < colsB; j++) {
        let sum = 0;
        for (let k = 0; k < colsA; k++) {
          const bValue = isMatrix ? (B[k] as number[])[j] : (B[k] as number);
          sum += A[i][k] * bValue;
        }
        result[i][j] = sum;
      }
    }
    
    return result;
  };
  
  // Simplified solver for linear system (this is a very basic implementation)
  const solveLinearSystem = (A: number[][], b: number[]) => {
    const n = A.length;
    const x = Array(n).fill(0);
    
    // Simple Gaussian elimination (not robust but works for our demo)
    for (let i = 0; i < n; i++) {
      let maxRow = i;
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(A[j][i]) > Math.abs(A[maxRow][i])) {
          maxRow = j;
        }
      }
      
      [A[i], A[maxRow]] = [A[maxRow], A[i]];
      [b[i], b[maxRow]] = [b[maxRow], b[i]];
      
      for (let j = i + 1; j < n; j++) {
        const factor = A[j][i] / A[i][i];
        b[j] -= factor * b[i];
        for (let k = i; k < n; k++) {
          A[j][k] -= factor * A[i][k];
        }
      }
    }
    
    // Back substitution
    for (let i = n - 1; i >= 0; i--) {
      let sum = 0;
      for (let j = i + 1; j < n; j++) {
        sum += A[i][j] * x[j];
      }
      x[i] = (b[i] - sum) / A[i][i];
    }
    
    return x;
  };
  
  // Handle lambda slider changes
  useEffect(() => {
    if (data.length > 0) {
      calculateRegressions(data, lambdaRidge[0], lambdaLasso[0]);
    }
  }, [lambdaRidge, lambdaLasso]);
  
  // Prepare visualization data
  const prepareVisualizationData = () => {
    return data.map((point, index) => {
      const x = point.x1; // Using x1 for visualization
      const y = point.y;
      
      const olsPrediction = coefficients.ols[0] + 
                           coefficients.ols[1] * point.x1 + 
                           coefficients.ols[2] * point.x2 + 
                           coefficients.ols[3] * point.x3;
      
      const ridgePrediction = coefficients.ridge[0] + 
                             coefficients.ridge[1] * point.x1 + 
                             coefficients.ridge[2] * point.x2 + 
                             coefficients.ridge[3] * point.x3;
      
      const lassoPrediction = coefficients.lasso[0] + 
                             coefficients.lasso[1] * point.x1 + 
                             coefficients.lasso[2] * point.x2 + 
                             coefficients.lasso[3] * point.x3;
      
      return {
        id: index,
        x,
        y,
        olsPrediction,
        ridgePrediction,
        lassoPrediction
      };
    }).sort((a, b) => a.x - b.x);
  };
  
  const visualData = prepareVisualizationData();
  
  // Get coefficient data for bar chart
  const getCoefficientsData = () => {
    return [
      { name: 'Intercept', ols: coefficients.ols[0], ridge: coefficients.ridge[0], lasso: coefficients.lasso[0] },
      { name: 'X1', ols: coefficients.ols[1], ridge: coefficients.ridge[1], lasso: coefficients.lasso[1] },
      { name: 'X2', ols: coefficients.ols[2], ridge: coefficients.ridge[2], lasso: coefficients.lasso[2] },
      { name: 'X3', ols: coefficients.ols[3], ridge: coefficients.ridge[3], lasso: coefficients.lasso[3] }
    ];
  };
  
  const coeffData = getCoefficientsData();
  
  // Calculate mean squared error
  const calculateMSE = (predictions: number[], actual: number[]) => {
    let sum = 0;
    for (let i = 0; i < predictions.length; i++) {
      sum += Math.pow(predictions[i] - actual[i], 2);
    }
    return (sum / predictions.length).toFixed(3);
  };
  
  const mseOLS = calculateMSE(
    data.map(point => coefficients.ols[0] + 
                     coefficients.ols[1] * point.x1 + 
                     coefficients.ols[2] * point.x2 + 
                     coefficients.ols[3] * point.x3),
    data.map(point => point.y)
  );
  
  const mseRidge = calculateMSE(
    data.map(point => coefficients.ridge[0] + 
                     coefficients.ridge[1] * point.x1 + 
                     coefficients.ridge[2] * point.x2 + 
                     coefficients.ridge[3] * point.x3),
    data.map(point => point.y)
  );
  
  const mseLasso = calculateMSE(
    data.map(point => coefficients.lasso[0] + 
                     coefficients.lasso[1] * point.x1 + 
                     coefficients.lasso[2] * point.x2 + 
                     coefficients.lasso[3] * point.x3),
    data.map(point => point.y)
  );

  return (
    <div className="p-4 space-y-6">
      <div className="mb-4">
        <h2 className="text-xl font-bold mb-2">Ridge vs Lasso Regression Simulator</h2>
        <p className="mb-4">Adjust the regularization strength (λ) to see how Ridge and Lasso regression affect the model coefficients.</p>
        
        <div className="flex flex-col gap-6 mb-4">
          <div className="flex gap-4">
            <button 
              className={`px-4 py-2 rounded-md ${regressionType === 'ridge' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
              onClick={() => setRegressionType('ridge')}
            >
              Ridge Regression
            </button>
            <button 
              className={`px-4 py-2 rounded-md ${regressionType === 'lasso' ? 'bg-green-600 text-white' : 'bg-gray-200'}`}
              onClick={() => setRegressionType('lasso')}
            >
              Lasso Regression
            </button>
          </div>
          
          {regressionType === 'ridge' ? (
            <div>
              <div className="flex items-center gap-2">
                <span className="min-w-16">Ridge λ:</span>
                <Slider 
                  value={lambdaRidge} 
                  onValueChange={setLambdaRidge} 
                  min={0} 
                  max={10} 
                  step={0.1} 
                  className="flex-grow"
                />
                <span className="min-w-12 text-right">{lambdaRidge[0].toFixed(1)}</span>
              </div>
              <p className="text-sm mt-1 text-gray-600">
                Ridge regression penalizes the sum of squared coefficient values (L2 norm). It shrinks coefficients toward zero but rarely makes them exactly zero.
              </p>
            </div>
          ) : (
            <div>
              <div className="flex items-center gap-2">
                <span className="min-w-16">Lasso λ:</span>
                <Slider 
                  value={lambdaLasso} 
                  onValueChange={setLambdaLasso} 
                  min={0} 
                  max={10} 
                  step={0.1} 
                  className="flex-grow"
                />
                <span className="min-w-12 text-right">{lambdaLasso[0].toFixed(1)}</span>
              </div>
              <p className="text-sm mt-1 text-gray-600">
                Lasso regression penalizes the sum of absolute coefficient values (L1 norm). It can shrink coefficients exactly to zero, performing feature selection.
              </p>
            </div>
          )}
        </div>
        
        <button 
          className="px-4 py-2 bg-gray-200 rounded-md hover:bg-gray-300"
          onClick={generateData}
        >
          Generate New Data
        </button>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="border p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-2">Coefficient Values</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart
                margin={{ top: 20, right: 30, left: 20, bottom: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="category" dataKey="name" name="Variable" />
                <YAxis type="number" name="Coefficient Value" />
                <Tooltip />
                <Legend />
                <Scatter name="OLS" data={coeffData} dataKey="ols" fill="#8884d8" />
                <Scatter name="Ridge" data={coeffData} dataKey="ridge" fill="#82ca9d" />
                <Scatter name="Lasso" data={coeffData} dataKey="lasso" fill="#ffc658" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2">
            <h4 className="font-semibold">Mean Squared Error (MSE):</h4>
            <ul className="text-sm">
              <li>OLS: {mseOLS}</li>
              <li>Ridge: {mseRidge}</li>
              <li>Lasso: {mseLasso}</li>
            </ul>
          </div>
        </div>
        
        <div className="border p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-2">Predictions vs Actual (X1 dimension)</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart
                margin={{ top: 20, right: 30, left: 20, bottom: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="x" name="X1" />
                <YAxis type="number" name="Y" />
                <Tooltip />
                <Legend />
                <Scatter name="Actual" data={visualData} dataKey="y" fill="#8884d8" />
                <Scatter name="OLS" data={visualData} dataKey="olsPrediction" fill="#82ca9d" line shape="cross" />
                <Scatter name="Ridge" data={visualData} dataKey="ridgePrediction" fill="#ffc658" line />
                <Scatter name="Lasso" data={visualData} dataKey="lassoPrediction" fill="#ff7300" line />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
      
      <div className="mt-4 p-4 bg-gray-100 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">Key Insights:</h3>
        <ul className="list-disc pl-5 space-y-2">
          <li>Ridge regression (L2 penalty) shrinks all coefficients toward zero but rarely makes them exactly zero</li>
          <li>Lasso regression (L1 penalty) can shrink coefficients exactly to zero, effectively performing feature selection</li>
          <li>As λ increases, regularization strength increases, reducing model complexity</li>
          <li>Notice how Ridge keeps the coefficient profile shape while Lasso tends to zero out less important features</li>
          <li>Both methods help prevent overfitting, especially with correlated features</li>
        </ul>
      </div>
    </div>
  );
};

export default RegressionSimulator; 