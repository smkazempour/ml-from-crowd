# Social Media Feature Proposals for Stock Prediction

**General Construction Rules:**
*   **Horizons:** standardized to $H \in \{5, 21, 63, 252\}$ trading days (approx 1 week, 1 month, 1 quarter, 1 year).
*   **No Look-ahead:** All calculations use data strictly available at time $t$.
*   **Date Mapping:** Leveraging Fama-French trading day counts where applicable.

## 1. Sentiment Dynamics (Momentum & Shifts)
Focusing on the *change* in opinion and the *direction* of that change.

1.  **Bullish Ratio**: $\frac{N_{Bull}}{N_{Bull} + N_{Bear}}$ using StockTwits user-provided labels.
2.  **Bearish Ratio**: $\frac{N_{Bear}}{N_{Bull} + N_{Bear}}$ using StockTwits user-provided labels.
3.  **Net Sentiment**: $\frac{N_{Bull} - N_{Bear}}{N_{Bull} + N_{Bear}}$ (ranges from -1 to +1).
4.  **Sentiment Delta**: Change in aggregate sentiment since $t-H$. ($S_t - S_{t-H}$).
5.  **Bear-to-Bull Flips (Count)**: Number of active users today who switched from Bearish (at most $H$ days ago) to Bullish today.
6.  **Bull-to-Bear Flips (Count)**: Number of active users today who switched from Bullish (at most $H$ days ago) to Bearish today.
7.  **Net Flip Count**: (Bear-to-Bull Flips) - (Bull-to-Bear Flips).
8.  **Bear-to-Bull Flip Ratio**: (Bear-to-Bull Flips) / (Total Active Users).
9.  **Bull-to-Bear Flip Ratio**: (Bull-to-Bear Flips) / (Total Active Users).
10. **Conviction Index**: % of users who have maintained the same sentiment direction for $>K$ consecutive tweets (where $K \in \{3, 5, 10\}$).
11. **Reversal Magnitude**: Net Sentiment$_t$ minus MovingAverage(Net Sentiment, $H$).
12. **Extreme Bullish Consensus**: Indicator if Bullish Ratio > 80% (or 90%).
13. **Extreme Bearish Consensus**: Indicator if Bearish Ratio > 80% (or 90%).
14. **After-Hours Sentiment/Volume**: Sentiment and Volume calculated specifically from Close ($t-1$) to Open ($t$).
15. **Market-Hours Sentiment/Volume**: Sentiment and Volume calculated from Open ($t$) to Close ($t$).
16. **First-Mover Sentiment**: Sentiment of the first $N$ users to post about this stock on day $t$ (where $N \in \{5, 10, 50\}$).

## 2. User Cohort Composition (Who is talking?)
Segmenting the crowd by experience and activity gaps.

17. **Fresh Blood Count**: Volume from users who have not tweeted about this stock in the last $H$ days (where $H \in \{21, 252, AllTime\}$).
18. **Fresh Blood Ratio**: (Fresh Blood Count) / (Total Volume).
19. **Re-entry Volume**: Volume from users inactive on this stock for $>H$ days but returned today.
20. **Whale Dominance**: % of volume from top 1% users based on cumulative tweet count up to $t-1$.
21. **Minnow Dominance**: % of volume from users with $<5$ tweets up to $t-1$.
22. **Night-Owl Ratio**: % of volume from users who post more than 80% of their tweets during After-Hours (rolling window classification).
23. **Day-Trader Ratio**: % of volume from users who post more than 80% of their tweets during Market-Hours (rolling window classification).
24. **Retention Rate**: % of users active today who posted at least one tweet about this stock in the previous $H$ days.
25. **Crowding Index**: Gini coefficient of user activity on day $t$ (high = few users dominating conversation; low = dispersed crowd).
26. **Specialist Ratio**: % of volume from users who tweet about $\leq 3$ unique stocks (deep knowledge vs. broad scanning).
27. **Sector Expert Ratio**: % of volume from users who primarily tweet about stocks in the same sector.

## 3. Volume & Attention Dynamics
Measuring raw activity and silence.

28. **Raw Volume**: Tweet count $V_t$ for stock $j$ on day $t$.
29. **Log Volume**: $\ln(1+V_t)$ (Handles zero volume days and reduces skewness).
30. **Unique User Count**: Number of distinct users posting about stock $j$ on day $t$.
31. **Volume Difference**: Raw tweet count $V_t$ minus $V_{t-1}$.
32. **Log Volume Change**: $\ln(1+V_t) - \ln(1+V_{t-1})$.
33. **Abnormal Volume (Simple)**: $V_t$ minus MovingAverage($V$, $H=21$).
34. **Abnormal Attention (Standardized)**: $\frac{V_t - \mu_H}{\sigma_H}$ where $\mu_H$ and $\sigma_H$ are rolling mean and standard deviation over $H$ days.
35. **Silence Gap**: Average time in days between consecutive tweets for stock $j$ over the past $H$ days (where $H \in \{21, 252\}$).
36. **Relative Volume**: Stock Volume$_t$ / Total Market Volume$_t$.
37. **Attention Surge**: Binary indicator if $V_t > \text{Mean}(V, H) + 2 \times \text{StdDev}(V, H)$.
38. **Attention Concentration (Herfindahl)**: $HHI_t = \sum_i s_{i,t}^2$ where $s_{i,t}$ is user $i$'s share of total volume on day $t$. High HHI = attention driven by few users; Low HHI = broad interest.

## 4. Network & Interaction Structure
Focusing on sentiment divergence.

39. **Disagreement Index**: $1 - \frac{|N_{Bull} - N_{Bear}|}{N_{Bull} + N_{Bear|}$.

## 5. Intraday & Calendar Coverage
Full 24-hour cycle coverage ensuring no time period is missing.

40. **Pre-Market Activity**: Volume/Sentiment from 04:00 to 09:30.
41. **Market-Open Frenzy**: Volume/Sentiment from 09:30 to 10:00.
42. **Late Morning**: Volume/Sentiment from 10:00 to 12:00.
43. **Mid-Day Lull**: Volume/Sentiment from 12:00 to 13:00.
44. **Early Afternoon**: Volume/Sentiment from 13:00 to 15:30.
45. **Closing Cross**: Volume/Sentiment from 15:30 to 16:00.
46. **Post-Market Activity**: Volume/Sentiment from 16:00 to 20:00.
47. **Overnight/Insomnia**: Volume/Sentiment from 20:00 to 04:00 (Next Day).
48. **Intraday Volatility (Sessions)**: Standard deviation of sentiment across the 8 defined daily sessions.

## 6. Symbol-Specific & Contextual
49. **Symbol Focus**: % of active users who tweeted *only* about this stock $j$ today.
50. **Co-mention Count (Complexity)**: Average number of *other* symbols mentioned alongside stock $j$ in today's messages. (High = correlation/sector trade; Low = specific news).
51. **Broadcasting/Spam Ratio**: % of messages about stock $j$ that mention $>5$ other stocks (Proxy for low-quality promotion).

*(Note: Total count is 51 high-quality definitions. Expansions via Horizons ($H$) will create additional variables.)*
