# fer_emotion_detection
A facial emotion model based on CNN, pre-trained in fer library.

# Emotion Analysis and Market Reaction

## Overview

This project explores the relationship between public emotions and market performance, specifically focusing on the SPY ETF around a significant financial event - the FOMC statement released on November 1, 2023. The analysis combines emotion data extracted from video content with SPY's intraday trading data to examine potential correlations and causal relationships.

## Dataset

Emotion Data (data_Nov.mp4.csv): Extracted emotions (e.g., anger, disgust, fear, sadness) from video content, presumably reflecting public sentiment at high-frequency intervals.
SPY Trading Data: Intraday trading data for the SPY ETF, obtained from the Alpha Vantage API, focusing on the timeframe around the FOMC statement release.

## Features

Emotion metrics: Aggregated and individual negative emotions.
Market data: OHLC (Open, High, Low, Close) and volume for SPY, with additional calculated features like minute-to-minute percentage change.
Time series analysis: Various statistical tests and models to explore correlations, causality, and potential predictive relationships.

## Analysis Highlights

Moving Average Convergence: Analysis of emotional indicators and market data using moving averages to identify trends.
Linear Regression: Assessed the impact of aggregated negative emotions on market movement.
Cross Correlation & Cointegration: Explored temporal relationships and potential long-term equilibrium between emotion indicators and market performance.
Dynamic Time Warping: Investigated the alignment and similarity between emotional and market time series.
Stationarity Tests: Checked the time series data for stationarity, a prerequisite for many time series analyses.
Granger Causality: Examined whether emotional indicators can predict future market movements.
