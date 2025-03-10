# Regression Simulator

An interactive web application for visualizing and comparing different types of regression models: Ordinary Least Squares (OLS), Ridge, and Lasso regression. Built with Next.js, TypeScript, and Recharts.

## Features

- **Interactive Visualization**: Compare OLS, Ridge, and Lasso regression models in real-time
- **Adjustable Parameters**: 
  - Control regularization strength (λ) for both Ridge and Lasso regression
  - Generate new random datasets with the click of a button
- **Multiple Visualizations**:
  - Coefficient comparison chart showing how different models affect feature weights
  - Scatter plot showing predictions vs actual values
- **Performance Metrics**: View Mean Squared Error (MSE) for each model
- **Educational Insights**: Learn about the differences between L1 and L2 regularization

## Getting Started

### Prerequisites

- Node.js 18.x or later
- npm 9.x or later

### Installation

1. Clone the repository:
```bash
git clone https://github.com/amanrai2508/Simulators.git
cd regression-simulator
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Usage

1. **Select Regression Type**:
   - Click on either "Ridge Regression" or "Lasso Regression" to focus on one model
   - Compare it with the baseline OLS model

2. **Adjust Regularization**:
   - Use the slider to control the regularization strength (λ)
   - Watch how the coefficients change in real-time
   - Observe the impact on model predictions

3. **Generate New Data**:
   - Click "Generate New Data" to create a new random dataset
   - See how different models perform on fresh data

4. **Interpret Results**:
   - Compare coefficient values across models
   - Observe how Ridge and Lasso handle feature selection differently
   - Check the MSE values to understand model performance

## Technical Details

### Implementation

- **Matrix Operations**: Custom implementation of matrix multiplication and linear system solving
- **Regression Algorithms**:
  - OLS: Standard least squares regression
  - Ridge: L2 regularization
  - Lasso: L1 regularization with coordinate descent

### Technologies Used

- Next.js 14
- TypeScript
- Recharts for data visualization
- Tailwind CSS for styling
- Shadcn UI components

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Next.js](https://nextjs.org/)
- Data visualization powered by [Recharts](https://recharts.org/)
- UI components from [shadcn/ui](https://ui.shadcn.com/) 