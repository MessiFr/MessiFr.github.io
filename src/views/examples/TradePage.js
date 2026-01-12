import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, Container, Grid, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TextField, MenuItem, Select, FormControl, InputLabel, Paper, Box, Typography, Chip, TablePagination, TableFooter } from '@mui/material';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

// Import navbar and footer components
import SolidNavbar from "components/Navbars/SolidNavbar";
import DefaultFooter from "components/Footers/DefaultFooter.js";

// Import Material UI Icons
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import AccountBalanceIcon from '@mui/icons-material/AccountBalance';
import EmojiEventsIcon from '@mui/icons-material/EmojiEvents';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import EqualizerIcon from '@mui/icons-material/Equalizer';
import AssessmentIcon from '@mui/icons-material/Assessment';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);


const TradePage = () => {
  const [chartData, setChartData] = useState(null);
  const [tableData, setTableData] = useState([]);
  const [summaryStats, setSummaryStats] = useState(null);
  const [conceptStageData, setConceptStageData] = useState([]);
  const [filteredConceptStageData, setFilteredConceptStageData] = useState([]);
  const [buySignalData, setBuySignalData] = useState([]); // New state for buy signals
  const [filteredBuySignalData, setFilteredBuySignalData] = useState([]);
  const [sellSignalData, setSellSignalData] = useState([]); // New state for sell signals
  const [filteredSellSignalData, setFilteredSellSignalData] = useState([]);
  const [dcNameFilter, setDcNameFilter] = useState('');
  const [stageFilter, setStageFilter] = useState('all');
  const [buySignalFilter, setBuySignalFilter] = useState({ // Changed to object for multiple filters
    date: 'all',
    ut_buy: 'all',
    longCond_base: 'all',
    stoch_buy_signal: 'all',
    fisher_buy_signal: 'all',
    long_confirm: 'all',
    over_ema5: 'all',
    score_min: 0,
    score_max: 100,
    dc_name: '',
  });
  const [sellSignalFilter, setSellSignalFilter] = useState({ // State for sell signal filters
    date: 'all',
    stock_name: '',
  });
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });
  
  // Pagination state for concept stage table
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  
  // Pagination state for buy signal table
  const [buySignalPage, setBuySignalPage] = useState(0);
  const [buySignalRowsPerPage, setBuySignalRowsPerPage] = useState(10);
  
  // Pagination state for sell signal table
  const [sellSignalPage, setSellSignalPage] = useState(0);
  const [sellSignalRowsPerPage, setSellSignalRowsPerPage] = useState(10);

  // Chart options
  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Portfolio Performance vs SSE (Cumulative Return %)',
        align: 'center'  // 标题居中
      },
    },
    scales: {
      y: {
        ticks: {
          callback: function(value) {
            return value.toFixed(2) + '%'; // Show percentage
          }
        },
        title: {
          display: true,
          text: 'Return (%)',
          align: 'center'  // y轴标题居中
        }
      },
      x: {
        title: {
          display: true,
          text: 'Date',
          align: 'center'  // x轴标题居中
        },
        ticks: {
          // 只显示部分标签，避免过于密集
          maxRotation: 45,  // 最大旋转角度
          minRotation: 0,   // 最小旋转角度
          maxTicksLimit: 10 // 限制最大标签数量
        }
      }
    }
  };

  // Handle sorting
  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  // Get sort indicator for table header
  const getSortIndicator = (columnName) => {
    if (sortConfig.key === columnName) {
      return sortConfig.direction === 'asc' ? ' ↑' : ' ↓';
    }
    return '';
  };

  useEffect(() => {
    // Fetch real data from API
    const fetchData = async () => {
      try {
        // Fetch portfolio data
        const portfolioResponse = await fetch('/api/portfolio-daily');
        const portfolioResult = await portfolioResponse.json();

        if (portfolioResult.success) {
          const data = portfolioResult.data;

          // Calculate summary statistics
          if (data.length > 0) {
            const latestRecord = [...data].sort((a, b) => b.date - a.date)[0];
            
            // Calculate summary stats
            // const initialCapital = Number(initialRecord.total_assets);
            const initialCapital = 100000;
            const finalAssets = Number(latestRecord.total_assets);
            const totalReturn = ((finalAssets - initialCapital) / initialCapital) * 100;
            const benchmarkReturn = parseFloat(latestRecord.index_return) * 100;
            const excessReturn = totalReturn - benchmarkReturn;
            
            setSummaryStats({
              initialCapital: '¥ ' + initialCapital.toLocaleString('zh-CN', { maximumFractionDigits: 2 }),
              finalAssets: '¥ ' + finalAssets.toLocaleString('zh-CN', { maximumFractionDigits: 2 }),
              totalReturn: totalReturn.toFixed(2) + '%',
              benchmarkReturn: benchmarkReturn.toFixed(2) + '%',
              excessReturn: excessReturn.toFixed(2) + '%'
            });
          }

          // Process data for chart - showing cumulative return over time
          const labels = data.map(item => item.date.toString()).reverse(); // Reverse to show chronological order
          const portfolioReturns = data.map(item => parseFloat(item.cum_return) * 100).reverse(); // Convert to percentage
          const benchmarkReturns = data.map(item => parseFloat(item.index_return) * 100).reverse(); // Convert to percentage

          // Prepare chart data
          const chartDataObj = {
            labels: labels,
            datasets: [
              {
                label: '策略 %',
                data: portfolioReturns,
                borderColor: '#3498db',  // 蓝色
                backgroundColor: 'rgba(52, 152, 219, 0.1)',  // 蓝色背景
                tension: 0.4,
                fill: true,
              },
              {
                label: '上证指数 %',
                data: benchmarkReturns,
                borderColor: '#e74c3c',  // 红色
                backgroundColor: 'rgba(231, 76, 60, 0.1)',  // 红色背景
                borderDash: [5, 5],
                tension: 0.4,
                fill: true,
              },
            ],
          };

          setChartData(chartDataObj);

          // Prepare table data (most recent entries)
          const tableRows = data.slice(0, 10).map(item => ({
            key: item.id || `${item.date}_${item.total_assets}`, // Adding key for Ant Design table
            date: item.date,
            portfolioValue: '¥ ' + Number(item.portfolio_value).toLocaleString('zh-CN', { maximumFractionDigits: 0 }),
            totalAssets: '¥ ' + Number(item.total_assets).toLocaleString('zh-CN', { maximumFractionDigits: 0 }),
            dailyReturn: '¥ ' + Number((Number(item.total_assets) * Number(item.daily_return)) / (Number(item.daily_return) + 1)).toLocaleString('zh-CN', { maximumFractionDigits: 0 }),
            dailyReturn1: (parseFloat(item.daily_return) * 100).toFixed(2) + '%',
            cumReturn: (parseFloat(item.cum_return) * 100).toFixed(2) + '%',
            holdings: item.holdings_count
          }))
          // .reverse(); // Reverse to show most recent first

          setTableData(tableRows);
        } else {
          console.error('Portfolio API request failed:', portfolioResult.message);
        }
        
        // Fetch concept stage data
        const conceptResponse = await fetch('/api/concept-stage');
        const conceptResult = await conceptResponse.json();

        console.log('conceptResponse')
        console.log(conceptResponse)

        if (conceptResult.success) {
          // Map the API response to include keys for the table
          const conceptData = conceptResult.data.map(item => ({
            key: item.dc_name, // Using dc_name as key for the table
            dc_name: item.dc_name,
            uplift: item.uplift,
            uplift_pct: item.uplift_pct,
            turnover_ratio: item.turnover_ratio,
            heat_score: item.heat_score,
            current_stage: item.current_stage,
            stage_desc: item.stage_desc
          }));
          
          setConceptStageData(conceptData);
          setFilteredConceptStageData(conceptData);
        } else {
          console.error('Concept stage API request failed:', conceptResult.message);
        }
        
        // Fetch buy signal data
        const buySignalResponse = await fetch('/api/lucy-buy-signal');
        const buySignalResult = await buySignalResponse.json();

        console.log('buySignalResponse')
        console.log(buySignalResponse)

        if (buySignalResult.success) {
          const buySignalData = buySignalResult.data.map(item => ({
            key: item.id, // Using id as key for the table
            id: item.id,
            trade_date: item.trade_date,
            stock_name: item.stock_name,
            score: item.score,
            ut_buy: item.ut_buy,
            longCond_base: item.longCond_base,
            stoch_buy_signal: item.stoch_buy_signal,
            fisher_buy_signal: item.fisher_buy_signal,
            long_confirm: item.long_confirm,
            over_ema5: item.over_ema5,
            superTrend_score_change: item.superTrend_score_change,
            dc_name: item.dc_name,
            heat_score: item.heat_score
          }));
          
          setBuySignalData(buySignalData);
          setFilteredBuySignalData(buySignalData);
        } else {
          console.error('Buy signals API request failed:', buySignalResult.message);
        }
        
        // Fetch sell signal data
        const sellSignalResponse = await fetch('/api/lucy-sell-signal/latest');
        const sellSignalResult = await sellSignalResponse.json();

        console.log('sellSignalResponse')
        console.log(sellSignalResponse)

        if (sellSignalResult.success) {
          const sellSignalData = sellSignalResult.data.map(item => ({
            key: `${item.trade_date}_${item.stock_name}`, // Using combination of trade_date and stock_name as key for the table
            trade_date: item.trade_date,
            stock_name: item.stock_name,
            ut_sell: item.ut_sell,
            shortCond_base: item.shortCond_base,
            stoch_sell_signal: item.stoch_sell_signal,
            fisher_sell_signal: item.fisher_sell_signal
          }));
          
          setSellSignalData(sellSignalData);
          setFilteredSellSignalData(sellSignalData);
        } else {
          console.error('Sell signals API request failed:', sellSignalResult.message);
        }
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, []);

  // Apply filters when filters change for concept stage table
  useEffect(() => {
    let result = [...conceptStageData];
    
    // Apply dc_name search filter (fuzzy search)
    if (dcNameFilter) {
      result = result.filter(item => 
        item.dc_name.toLowerCase().includes(dcNameFilter.toLowerCase())
      );
    }
    
    // Apply current_stage filter
    if (stageFilter !== 'all') {
      result = result.filter(item => item.current_stage === stageFilter);
    }
    
    // Apply sorting
    if (sortConfig.key) {
      result.sort((a, b) => {
        // @ts-ignore
        if (a[sortConfig.key] < b[sortConfig.key]) {
          return sortConfig.direction === 'asc' ? -1 : 1;
        }
        // @ts-ignore
        if (a[sortConfig.key] > b[sortConfig.key]) {
          return sortConfig.direction === 'asc' ? 1 : -1;
        }
        return 0;
      });
    }
    
    setFilteredConceptStageData(result);
    // Reset to first page when filters change
    setPage(0);
  }, [dcNameFilter, stageFilter, sortConfig, conceptStageData]);
  
  // Apply filters when filters change for buy signal table
  useEffect(() => {
    let result = [...buySignalData];
    
    // Apply date filter
    if (buySignalFilter.date !== 'all') {
      result = result.filter(item => item.trade_date === buySignalFilter.date);
    }
    
    // Apply score range filter
    result = result.filter(item => 
      parseFloat(item.score) >= buySignalFilter.score_min && 
      parseFloat(item.score) <= buySignalFilter.score_max
    );
    
    // Apply UT buy signal filter
    if (buySignalFilter.ut_buy !== 'all') {
      result = result.filter(item => String(item.ut_buy) === buySignalFilter.ut_buy);
    }
    
    // Apply long condition filter
    if (buySignalFilter.longCond_base !== 'all') {
      result = result.filter(item => String(item.longCond_base) === buySignalFilter.longCond_base);
    }
    
    // Apply stochastic buy signal filter
    if (buySignalFilter.stoch_buy_signal !== 'all') {
      result = result.filter(item => String(item.stoch_buy_signal) === buySignalFilter.stoch_buy_signal);
    }
    
    // Apply fisher buy signal filter
    if (buySignalFilter.fisher_buy_signal !== 'all') {
      result = result.filter(item => String(item.fisher_buy_signal) === buySignalFilter.fisher_buy_signal);
    }
    
    // Apply long confirm filter
    if (buySignalFilter.long_confirm !== 'all') {
      result = result.filter(item => String(item.long_confirm) === buySignalFilter.long_confirm);
    }
    
    // Apply over EMA5 filter
    if (buySignalFilter.over_ema5 !== 'all') {
      result = result.filter(item => String(item.over_ema5) === buySignalFilter.over_ema5);
    }
    
    // Apply DC name filter (fuzzy search)
    if (buySignalFilter.dc_name) {
      result = result.filter(item => 
        item.dc_name.toLowerCase().includes(buySignalFilter.dc_name.toLowerCase())
      );
    }
    
    setFilteredBuySignalData(result);
    // Reset to first page when filters change
    setBuySignalPage(0);
  }, [buySignalFilter, buySignalData]);

  // Apply filters when filters change for sell signal table
  useEffect(() => {
    let result = [...sellSignalData];
    
    // Apply date filter
    if (sellSignalFilter.date !== 'all') {
      result = result.filter(item => item.trade_date === sellSignalFilter.date);
    }
    
    // Apply stock name filter (fuzzy search)
    if (sellSignalFilter.stock_name) {
      result = result.filter(item => 
        item.stock_name.toLowerCase().includes(sellSignalFilter.stock_name.toLowerCase())
      );
    }
    
    setFilteredSellSignalData(result);
    // Reset to first page when filters change
    setSellSignalPage(0);
  }, [sellSignalFilter, sellSignalData]);
  
  // Get unique stages for dropdown
  const getUniqueStages = () => {
    const stages = [...new Set(conceptStageData.map(item => item.current_stage))];
    return stages;
  };
  
  // Get unique dates for buy signal filter
  const getUniqueDates = () => {
    const dates = [...new Set(buySignalData.map(item => item.trade_date))];
    return dates.sort((a, b) => new Date(b) - new Date(a)); // Sort by date descending
  };
  
  // Get unique dates for sell signal filter
  const getUniqueSellDates = () => {
    const dates = [...new Set(sellSignalData.map(item => item.trade_date))];
    return dates.sort((a, b) => new Date(b) - new Date(a)); // Sort by date descending
  };
  
  // Determine chip color based on stage
  const getStageChipColor = (stage) => {
    if (stage.includes('启动') || stage.includes('初期')) return 'primary';
    else if (stage.includes('主升') || stage.includes('成长')) return 'success';
    else if (stage.includes('过热') || stage.includes('高位')) return 'warning';
    else if (stage.includes('退潮') || stage.includes('回落')) return 'error';
    return 'default';
  };

  // Handle pagination change
  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  // Handle pagination change for buy signal table
  const handleBuySignalChangePage = (event, newPage) => {
    setBuySignalPage(newPage);
  };

  const handleBuySignalChangeRowsPerPage = (event) => {
    setBuySignalRowsPerPage(parseInt(event.target.value, 10));
    setBuySignalPage(0);
  };
  
  // Handle pagination change for sell signal table
  const handleSellSignalChangePage = (event, newPage) => {
    setSellSignalPage(newPage);
  };

  const handleSellSignalChangeRowsPerPage = (event) => {
    setSellSignalRowsPerPage(parseInt(event.target.value, 10));
    setSellSignalPage(0);
  };
  
  // Helper function to safely format numbers
  const safeFormatNumber = (value, decimals = 2, isPercentage = false) => {
    if (value === null || value === undefined || value === '') {
      return 'N/A';
    }
    
    let numValue;
    if (typeof value === 'number') {
      numValue = value;
    } else if (typeof value === 'string') {
      numValue = parseFloat(value);
      if (isNaN(numValue)) {
        return 'N/A';
      }
    } else {
      return 'N/A';
    }
    
    if (isPercentage) {
      return (numValue * 100).toFixed(decimals) + '%';
    } else {
      return numValue.toFixed(decimals);
    }
  };


  return (
    <>
      <SolidNavbar label="Trade Page"/>
      <div className="wrapper">
        <div className="section section-hero section-shaped">
          <div className="shape shape-style-1 shape-default">
            <span className="span-150" />
            <span className="span-50" />
            <span className="span-50" />
            <span className="span-75" />
            <span className="span-100" />
            <span className="span-75" />
            <span className="span-50" />
            <span className="span-100" />
            <span className="span-50" />
            <span className="span-100" />
          </div>
        </div>
        <section className="section section-components pb-0">
          <Container maxWidth="xl" sx={{ py: 4 }}>
            <Grid container spacing={3} justifyContent="center">
              <Grid item xs={12}>
                <Box display="flex" alignItems="center" mb={4}>
                  <AssessmentIcon sx={{ fontSize: 32, mr: 1, color: '#1052cc' }} />
                  <Typography variant="h3" component="h2">交易日记</Typography>
                </Box>
              </Grid>
            
            {/* Summary Statistics in a single row */}
            {summaryStats && (
              <Grid item xs={12}>
                <Paper elevation={3} sx={{ p: 3, mb: 4, backgroundColor: '#f0f2f5' }}>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={2.4}>
                      <Box textAlign="center">
                        <Box display="flex" alignItems="center" justifyContent="center" mb={1}>
                          <AccountBalanceIcon sx={{ fontSize: 20, mr: 1, color: '#666' }} />
                          <Typography variant="subtitle2" color="textSecondary">初始本金</Typography>
                        </Box>
                        <Typography variant="h6" fontWeight="bold" color="#333">{summaryStats.initialCapital}</Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={12} md={2.4}>
                      <Box textAlign="center">
                        <Box display="flex" alignItems="center" justifyContent="center" mb={1}>
                          <TrendingUpIcon sx={{ fontSize: 20, mr: 1, color: '#666' }} />
                          <Typography variant="subtitle2" color="textSecondary">最终总资产</Typography>
                        </Box>
                        <Typography variant="h6" fontWeight="bold" color="#333">{summaryStats.finalAssets}</Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={12} md={2.4}>
                      <Box textAlign="center">
                        <Box display="flex" alignItems="center" justifyContent="center" mb={1}>
                          <ShowChartIcon sx={{ fontSize: 20, mr: 1, color: '#666' }} />
                          <Typography variant="subtitle2" color="textSecondary">总收益率</Typography>
                        </Box>
                        <Typography variant="h6" fontWeight="bold" color="#333">{summaryStats.totalReturn}</Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={12} md={2.4}>
                      <Box textAlign="center">
                        <Box display="flex" alignItems="center" justifyContent="center" mb={1}>
                          <EqualizerIcon sx={{ fontSize: 20, mr: 1, color: '#666' }} />
                          <Typography variant="subtitle2" color="textSecondary">上证指数收益率</Typography>
                        </Box>
                        <Typography variant="h6" fontWeight="bold" color="#333">{summaryStats.benchmarkReturn}</Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={12} md={2.4}>
                      <Box textAlign="center">
                        <Box display="flex" alignItems="center" justifyContent="center" mb={1}>
                          <EmojiEventsIcon sx={{ fontSize: 20, mr: 1, color: '#666' }} />
                          <Typography variant="subtitle2" color="textSecondary">超额收益</Typography>
                        </Box>
                        <Typography variant="h6" fontWeight="bold" color="#333">{summaryStats.excessReturn}</Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </Paper>
              </Grid>
            )}

            {/* Chart Section */}
            <Grid item xs={12}>
                <Card elevation={3} sx={{ mb: 4 }}>
                  <CardHeader 
                    title={
                      <Box textAlign="center" pt={1}>
                        <Typography variant="h5" component="div">策略收益 vs 上证指数</Typography>
                      </Box>
                    } 
                  />
                  <CardContent sx={{ height: '800px' }}>
                    {chartData ? (
                      <Box sx={{ height: '100%', width: '100%' }}>
                        <Line data={chartData} options={options} />
                      </Box>
                    ) : (
                      <Typography variant="body1">Loading chart data...</Typography>
                    )}
                  </CardContent>
                </Card>
              </Grid>

            {/* Trading History Table */}
            <Grid item xs={12}>
                <Card elevation={3} sx={{ mb: 4 }}>
                  <CardHeader title={
                    <Typography variant="h5" component="div">近期收益</Typography>
                  } />
                  <CardContent>
                    <TableContainer>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>日期</TableCell>
                            <TableCell>总资产</TableCell>
                            <TableCell>仓位</TableCell>
                            <TableCell>日收益</TableCell>
                            <TableCell>日收益率</TableCell>
                            <TableCell>累计收益率</TableCell>
                            <TableCell>持仓数</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {tableData.map((row) => (
                            <TableRow key={row.key}>
                              <TableCell>{row.date}</TableCell>
                              <TableCell>{row.totalAssets}</TableCell>
                              <TableCell>{row.portfolioValue}</TableCell>
                              <TableCell>{row.dailyReturn}</TableCell>
                              <TableCell>{row.dailyReturn1}</TableCell>
                              <TableCell>{row.cumReturn}</TableCell>
                              <TableCell>{row.holdings}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </CardContent>
                </Card>
              </Grid>

            {/* Concept Stage Table with Filters */}
            <Grid item xs={12}>
                <Card elevation={3}>
                  <CardHeader 
                    title={
                      <Typography variant="h5" component="div">概念板块分析</Typography>
                    } 
                  />
                  <CardContent>
                    {/* Filter Controls */}
                    <Box mb={3} display="flex" gap={2} flexWrap="wrap">
                      <TextField
                        label="概念板块名称搜索"
                        variant="outlined"
                        fullWidth={false}
                        value={dcNameFilter}
                        onChange={(e) => setDcNameFilter(e.target.value)}
                        sx={{ minWidth: 250 }}
                        placeholder="输入关键词搜索..."
                      />
                      <FormControl sx={{ minWidth: 200 }}>
                        <InputLabel>当前阶段</InputLabel>
                        <Select
                          value={stageFilter}
                          onChange={(e) => setStageFilter(e.target.value)}
                          label="当前阶段"
                        >
                          <MenuItem value="all">全部阶段</MenuItem>
                          {getUniqueStages().map((stage, index) => (
                            <MenuItem key={index} value={stage}>{stage}</MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Box>
                    
                    <TableContainer component={Paper}>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell onClick={() => handleSort('dc_name')} style={{ cursor: 'pointer' }}>
                              概念板块{getSortIndicator('dc_name')}
                            </TableCell>
                            <TableCell onClick={() => handleSort('uplift')} style={{ cursor: 'pointer' }}>
                              提振{getSortIndicator('uplift')}
                            </TableCell>
                            <TableCell onClick={() => handleSort('uplift_pct')} style={{ cursor: 'pointer' }}>
                              提振百分比{getSortIndicator('uplift_pct')}
                            </TableCell>
                            <TableCell onClick={() => handleSort('turnover_ratio')} style={{ cursor: 'pointer' }}>
                              换手率{getSortIndicator('turnover_ratio')}
                            </TableCell>
                            <TableCell onClick={() => handleSort('heat_score')} style={{ cursor: 'pointer' }}>
                              热度评分{getSortIndicator('heat_score')}
                            </TableCell>
                            <TableCell onClick={() => handleSort('current_stage')} style={{ cursor: 'pointer' }}>
                              当前阶段{getSortIndicator('current_stage')}
                            </TableCell>
                            <TableCell onClick={() => handleSort('stage_desc')} style={{ cursor: 'pointer' }}>
                              阶段描述{getSortIndicator('stage_desc')}
                            </TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {filteredConceptStageData.length > 0 ? (
                            filteredConceptStageData
                              .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                              .map((item) => (
                                <TableRow key={item.key}>
                                  <TableCell><Typography component="a" href="#" color="primary">{item.dc_name}</Typography></TableCell>
                                  <TableCell>{safeFormatNumber(item.uplift, 2) || 'N/A'}</TableCell>
                                  <TableCell>{safeFormatNumber(item.uplift_pct / 100, 2, true)}</TableCell>
                                  <TableCell>{safeFormatNumber(item.turnover_ratio / 100, 2, true)}</TableCell>
                                  <TableCell>{safeFormatNumber(item.heat_score, 2)}</TableCell>
                                  <TableCell>
                                    <Chip 
                                      label={item.current_stage} 
                                      color={getStageChipColor(item.current_stage)} 
                                      size="small" 
                                    />
                                  </TableCell>
                                  <TableCell>{item.stage_desc}</TableCell>
                                </TableRow>
                              ))
                          ) : (
                            <TableRow>
                              <TableCell colSpan={7} align="center">没有找到匹配的数据</TableCell>
                            </TableRow>
                          )}
                        </TableBody>
                        <TableFooter>
                          <TableRow>
                            <TablePagination
                              rowsPerPageOptions={[5, 10, 25]}
                              colSpan={7}
                              count={filteredConceptStageData.length}
                              rowsPerPage={rowsPerPage}
                              page={page}
                              onPageChange={handleChangePage}
                              onRowsPerPageChange={handleChangeRowsPerPage}
                              labelRowsPerPage="每页行数"
                              labelDisplayedRows={({ from, to, count }) => {
                                return `${from}-${to} 共 ${count} 条记录`;
                              }}
                            />
                          </TableRow>
                        </TableFooter>
                      </Table>
                    </TableContainer>
                  </CardContent>
                </Card>
              </Grid>
              
              {/* Buy Signal Table */}
              <Grid item xs={12}>
                <Card elevation={3} sx={{ mt: 4 }}>
                  <CardHeader 
                    title={
                      <Box display="flex" alignItems="center">
                        <TrendingUpIcon sx={{ mr: 1, color: '#1052cc' }} />
                        <Typography variant="h5" component="div">买入信号推荐</Typography>
                      </Box>
                    } 
                  />
                  <CardContent>
                    {/* Filter Controls for Buy Signals */}
                    <Box mb={3} display="flex" gap={2} flexWrap="wrap">
                      <FormControl sx={{ minWidth: 150 }}>
                        <InputLabel>交易日期</InputLabel>
                        <Select
                          value={buySignalFilter.date}
                          onChange={(e) => setBuySignalFilter({...buySignalFilter, date: e.target.value})}
                          label="交易日期"
                        >
                          <MenuItem value="all">全部日期</MenuItem>
                          {getUniqueDates().map((date, index) => (
                            <MenuItem key={index} value={date}>{date}</MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                      
                      <TextField
                        label="综合评分范围(最小)"
                        variant="outlined"
                        type="number"
                        value={buySignalFilter.score_min}
                        onChange={(e) => setBuySignalFilter({...buySignalFilter, score_min: parseFloat(e.target.value) || 0})}
                        sx={{ minWidth: 120 }}
                      />
                      
                      <TextField
                        label="综合评分范围(最大)"
                        variant="outlined"
                        type="number"
                        value={buySignalFilter.score_max}
                        onChange={(e) => setBuySignalFilter({...buySignalFilter, score_max: parseFloat(e.target.value) || 100})}
                        sx={{ minWidth: 120 }}
                      />
                      
                      <FormControl sx={{ minWidth: 150 }}>
                        <InputLabel>UT买信号</InputLabel>
                        <Select
                          value={buySignalFilter.ut_buy}
                          onChange={(e) => setBuySignalFilter({...buySignalFilter, ut_buy: e.target.value})}
                          label="UT买信号"
                        >
                          <MenuItem value="all">全部</MenuItem>
                          <MenuItem value="true">是</MenuItem>
                          <MenuItem value="false">否</MenuItem>
                        </Select>
                      </FormControl>
                      
                      <FormControl sx={{ minWidth: 150 }}>
                        <InputLabel>做多条件确认</InputLabel>
                        <Select
                          value={buySignalFilter.longCond_base}
                          onChange={(e) => setBuySignalFilter({...buySignalFilter, longCond_base: e.target.value})}
                          label="做多条件确认"
                        >
                          <MenuItem value="all">全部</MenuItem>
                          <MenuItem value="true">是</MenuItem>
                          <MenuItem value="false">否</MenuItem>
                        </Select>
                      </FormControl>
                      
                      <FormControl sx={{ minWidth: 150 }}>
                        <InputLabel>StochRSI买信号</InputLabel>
                        <Select
                          value={buySignalFilter.stoch_buy_signal}
                          onChange={(e) => setBuySignalFilter({...buySignalFilter, stoch_buy_signal: e.target.value})}
                          label="StochRSI买信号"
                        >
                          <MenuItem value="all">全部</MenuItem>
                          <MenuItem value="true">是</MenuItem>
                          <MenuItem value="false">否</MenuItem>
                        </Select>
                      </FormControl>
                      
                      <FormControl sx={{ minWidth: 150 }}>
                        <InputLabel>费舍尔买信号</InputLabel>
                        <Select
                          value={buySignalFilter.fisher_buy_signal}
                          onChange={(e) => setBuySignalFilter({...buySignalFilter, fisher_buy_signal: e.target.value})}
                          label="费舍尔买信号"
                        >
                          <MenuItem value="all">全部</MenuItem>
                          <MenuItem value="true">是</MenuItem>
                          <MenuItem value="false">否</MenuItem>
                        </Select>
                      </FormControl>
                      
                      <FormControl sx={{ minWidth: 150 }}>
                        <InputLabel>长期确认</InputLabel>
                        <Select
                          value={buySignalFilter.long_confirm}
                          onChange={(e) => setBuySignalFilter({...buySignalFilter, long_confirm: e.target.value})}
                          label="长期确认"
                        >
                          <MenuItem value="all">全部</MenuItem>
                          <MenuItem value="true">是</MenuItem>
                          <MenuItem value="false">否</MenuItem>
                        </Select>
                      </FormControl>
                      
                      <FormControl sx={{ minWidth: 150 }}>
                        <InputLabel>超EMA5</InputLabel>
                        <Select
                          value={buySignalFilter.over_ema5}
                          onChange={(e) => setBuySignalFilter({...buySignalFilter, over_ema5: e.target.value})}
                          label="超EMA5"
                        >
                          <MenuItem value="all">全部</MenuItem>
                          <MenuItem value="true">是</MenuItem>
                          <MenuItem value="false">否</MenuItem>
                        </Select>
                      </FormControl>
                      
                      <TextField
                        label="概念板块"
                        variant="outlined"
                        value={buySignalFilter.dc_name}
                        onChange={(e) => setBuySignalFilter({...buySignalFilter, dc_name: e.target.value})}
                        sx={{ minWidth: 150 }}
                        placeholder="输入关键词..."
                      />
                    </Box>
                    
                    <TableContainer component={Paper}>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>股票名称</TableCell>
                            <TableCell>交易日期</TableCell>
                            <TableCell>综合得分</TableCell>
                            <TableCell>UT买信号</TableCell>
                            <TableCell>做多条件确认</TableCell>
                            <TableCell>StochRSI买信号</TableCell>
                            <TableCell>费舍尔买信号</TableCell>
                            <TableCell>长期确认</TableCell>
                            <TableCell>超EMA5</TableCell>
                            <TableCell>超级趋势变化</TableCell>
                            <TableCell>概念板块</TableCell>
                            <TableCell>热度评分</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {filteredBuySignalData.length > 0 ? (
                            filteredBuySignalData
                              .slice(buySignalPage * buySignalRowsPerPage, buySignalPage * buySignalRowsPerPage + buySignalRowsPerPage)
                              .map((item) => (
                                <TableRow key={item.key}>
                                  <TableCell><strong>{item.stock_name}</strong></TableCell>
                                  <TableCell>{item.trade_date}</TableCell>
                                  <TableCell><Chip label={Math.round(item.score)} color="primary" size="small" /></TableCell>
                                  <TableCell>
                                    <Chip 
                                      label={item.ut_buy ? '是' : '否'} 
                                      color={item.ut_buy ? 'success' : 'default'} 
                                      size="small" 
                                    />
                                  </TableCell>
                                  <TableCell>
                                    <Chip 
                                      label={item.longCond_base ? '是' : '否'} 
                                      color={item.longCond_base ? 'success' : 'default'} 
                                      size="small" 
                                    />
                                  </TableCell>
                                  <TableCell>
                                    <Chip 
                                      label={item.stoch_buy_signal ? '是' : '否'} 
                                      color={item.stoch_buy_signal ? 'success' : 'default'} 
                                      size="small" 
                                    />
                                  </TableCell>
                                  <TableCell>
                                    <Chip 
                                      label={item.fisher_buy_signal ? '是' : '否'} 
                                      color={item.fisher_buy_signal ? 'success' : 'default'} 
                                      size="small" 
                                    />
                                  </TableCell>
                                  <TableCell>
                                    <Chip 
                                      label={item.long_confirm ? '是' : '否'} 
                                      color={item.long_confirm ? 'primary' : 'default'} 
                                      size="small" 
                                    />
                                  </TableCell>
                                  <TableCell>
                                    <Chip 
                                      label={item.over_ema5 ? '是' : '否'} 
                                      color={item.over_ema5 ? 'primary' : 'default'} 
                                      size="small" 
                                    />
                                  </TableCell>
                                  <TableCell>{item.superTrend_score_change !== null && item.superTrend_score_change !== undefined ? Math.round(item.superTrend_score_change) : 'N/A'}</TableCell>
                                  <TableCell>{item.dc_name || '-'}</TableCell>
                                  <TableCell>{safeFormatNumber(item.heat_score, 2)}</TableCell>
                                </TableRow>
                              ))
                          ) : (
                            <TableRow>
                              <TableCell colSpan={12} align="center">没有找到匹配的买入信号</TableCell>
                            </TableRow>
                          )}
                        </TableBody>
                        <TableFooter>
                          <TableRow>
                            <TablePagination
                              rowsPerPageOptions={[5, 10, 25]}
                              colSpan={12}
                              count={filteredBuySignalData.length}
                              rowsPerPage={buySignalRowsPerPage}
                              page={buySignalPage}
                              onPageChange={handleBuySignalChangePage}
                              onRowsPerPageChange={handleBuySignalChangeRowsPerPage}
                              labelRowsPerPage="每页行数"
                              labelDisplayedRows={({ from, to, count }) => {
                                return `${from}-${to} 共 ${count} 条记录`;
                              }}
                            />
                          </TableRow>
                        </TableFooter>
                      </Table>
                    </TableContainer>
                  </CardContent>
                </Card>
              </Grid>
              
              {/* Sell Signal Table */}
              <Grid item xs={12}>
                <Card elevation={3} sx={{ mt: 4 }}>
                  <CardHeader 
                    title={
                      <Box display="flex" alignItems="center">
                        <TrendingUpIcon sx={{ mr: 1, color: '#e74c3c' }} />
                        <Typography variant="h5" component="div">卖出信号推荐</Typography>
                      </Box>
                    } 
                  />
                  <CardContent>
                    {/* Filter Controls for Sell Signals */}
                    <Box mb={3} display="flex" gap={2} flexWrap="wrap">
                      <FormControl sx={{ minWidth: 150 }}>
                        <InputLabel>交易日期</InputLabel>
                        <Select
                          value={sellSignalFilter.date}
                          onChange={(e) => setSellSignalFilter({...sellSignalFilter, date: e.target.value})}
                          label="交易日期"
                        >
                          <MenuItem value="all">全部日期</MenuItem>
                          {getUniqueSellDates().map((date, index) => (
                            <MenuItem key={index} value={date}>{date}</MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                      
                      <TextField
                        label="股票名称"
                        variant="outlined"
                        value={sellSignalFilter.stock_name}
                        onChange={(e) => setSellSignalFilter({...sellSignalFilter, stock_name: e.target.value})}
                        sx={{ minWidth: 150 }}
                        placeholder="输入股票名称..."
                      />
                    </Box>
                    
                    <TableContainer component={Paper}>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>股票名称</TableCell>
                            <TableCell>交易日期</TableCell>
                            <TableCell>UT卖信号</TableCell>
                            <TableCell>短期条件</TableCell>
                            <TableCell>随机卖信号</TableCell>
                            <TableCell>费舍尔卖信号</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {filteredSellSignalData.length > 0 ? (
                            filteredSellSignalData
                              .slice(sellSignalPage * sellSignalRowsPerPage, sellSignalPage * sellSignalRowsPerPage + sellSignalRowsPerPage)
                              .map((item) => (
                                <TableRow key={item.key}>
                                  <TableCell><strong>{item.stock_name}</strong></TableCell>
                                  <TableCell>{item.trade_date}</TableCell>
                                  <TableCell>
                                    <Chip 
                                      label={item.ut_sell ? '是' : '否'} 
                                      color={item.ut_sell ? 'error' : 'default'} 
                                      size="small" 
                                    />
                                  </TableCell>
                                  <TableCell>
                                    <Chip 
                                      label={item.shortCond_base ? '是' : '否'} 
                                      color={item.shortCond_base ? 'error' : 'default'} 
                                      size="small" 
                                    />
                                  </TableCell>
                                  <TableCell>
                                    <Chip 
                                      label={item.stoch_sell_signal ? '是' : '否'} 
                                      color={item.stoch_sell_signal ? 'error' : 'default'} 
                                      size="small" 
                                    />
                                  </TableCell>
                                  <TableCell>
                                    <Chip 
                                      label={item.fisher_sell_signal ? '是' : '否'} 
                                      color={item.fisher_sell_signal ? 'error' : 'default'} 
                                      size="small" 
                                    />
                                  </TableCell>
                                </TableRow>
                              ))
                          ) : (
                            <TableRow>
                              <TableCell colSpan={6} align="center">没有找到匹配的卖出信号</TableCell>
                            </TableRow>
                          )}
                        </TableBody>
                        <TableFooter>
                          <TableRow>
                            <TablePagination
                              rowsPerPageOptions={[5, 10, 25]}
                              colSpan={6}
                              count={filteredSellSignalData.length}
                              rowsPerPage={sellSignalRowsPerPage}
                              page={sellSignalPage}
                              onPageChange={handleSellSignalChangePage}
                              onRowsPerPageChange={handleSellSignalChangeRowsPerPage}
                              labelRowsPerPage="每页行数"
                              labelDisplayedRows={({ from, to, count }) => {
                                return `${from}-${to} 共 ${count} 条记录`;
                              }}
                            />
                          </TableRow>
                        </TableFooter>
                      </Table>
                    </TableContainer>
                  </CardContent>
                </Card>
              </Grid>
          </Grid>
          </Container>
        </section>
        <DefaultFooter />
      </div>
    </>
  );
};

export default TradePage;