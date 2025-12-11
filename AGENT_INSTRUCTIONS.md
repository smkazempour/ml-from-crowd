# Agent Instructions for Financial Social Media ML Project

## Project Overview
This project applies machine learning techniques to extract signals from financial social media data. The goal is to discover new features beyond traditional sentiment and attention measures that can predict stock returns, using modern ML methods rather than conventional linear regression approaches.

## Core Principles

### 1. Data Management
- **Memory Efficiency**: Work with large datasets without creating unnecessary copies
- **In-place Operations**: Prefer modifications that don't duplicate data
- **Careful Loading**: Only load required columns/rows when possible
- **No Redundancy**: Avoid saving intermediate versions unless explicitly requested

### 2. Code Organization
- **Minimal Clutter**: Keep code clean and focused
- **Follow Instructions Precisely**: Stay on task unless creative exploration is explicitly requested
- **Structured Notebooks**: Organize cells in a consistent pattern:
  1. Import statements (first 1-2 cells)
  2. Directory definitions and global variables (next 2-3 cells)
  3. Analysis code (remaining cells)

### 3. Naming Conventions
- **Global Variables**: Use CAPITAL_LETTERS for:
  - Directory paths (e.g., `DATA_DIR`, `FIGURES_DIR`, `TABLES_DIR`, `CODE_DIR`)
  - File names (e.g., `INPUT_FILE`, `OUTPUT_FILE`)
  - Parameters (e.g., `WINDOW_SIZE`, `THRESHOLD`, `N_ESTIMATORS`)
- **Local Variables**: Use lowercase_with_underscores

### 4. Project Structure
- **CODE_DIR**: Current working directory (Code/) - Contains analysis notebooks and scripts
- **DATA_DIR**: Separate location - Raw and processed datasets
- **FIGURES_DIR**: Adjacent to Code/ - Visualizations and plots
- **TABLES_DIR**: Adjacent to Code/ - Results tables and summaries

## Response Guidelines

### When to Be Creative
- User explicitly asks to "explore" or "investigate"
- User requests "new ways" to analyze data
- User asks for suggestions or recommendations
- User wants to discover patterns or features

### When to Stay Focused
- User provides specific instructions
- User asks for a particular analysis or visualization
- User requests implementation of a known method
- User specifies exact output format

### Output Expectations
- **Visualizations**: Create clear, publication-quality figures
- **Tables**: Present results in well-formatted, interpretable tables
- **Code**: Write efficient, documented, reproducible code
- **Results**: Provide quantitative summaries and insights

## Technical Requirements

### Package Management
- Import all packages at the top of notebooks
- Group imports logically (standard library, data science, ML, visualization)
- Document any specialized or uncommon packages

### Analysis Approach
- **Feature Engineering**: Beyond sentiment and attention
- **ML Methods**: Random forests, gradient boosting, neural networks, etc.
- **Baselines**: Compare to linear regression benchmarks
- **Validation**: Proper train/test splits, cross-validation for financial time series

### Documentation
- Comment complex operations
- Explain feature engineering decisions
- Document model choices and hyperparameters
- Provide interpretation of results

## Workflow Expectations

1. **Step-by-Step Collaboration**: Work incrementally with user guidance
2. **Confirmation**: Ask for clarification when instructions are ambiguous
3. **Efficiency**: Minimize unnecessary operations and file I/O
4. **Reproducibility**: Ensure code can be re-run with consistent results
5. **Communication**: Provide brief, clear updates on progress

## Key Metrics and Concepts

### Traditional Social Media Measures
- **Sentiment**: Bullish vs. bearish classification
- **Attention**: Volume of messages/tweets per stock per day

### Project Goals
- Extract novel features using ML techniques
- Evaluate predictive power for stock returns
- Compare ML approaches to traditional linear methods
- Demonstrate value-add beyond standard sentiment/attention metrics

## Quality Standards
- Code should be production-quality, not experimental
- Visualizations should be publication-ready
- Results should be statistically rigorous
- Analysis should be computationally efficient given large datasets

---

**Last Updated**: December 3, 2025
