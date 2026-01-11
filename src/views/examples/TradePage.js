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

// 定义数据类型
const TradePage = () => {
  const [chartData, setChartData] = useState(null);
  const [tableData, setTableData] = useState([]);
  const [summaryStats, setSummaryStats] = useState(null);
  const [conceptStageData, setConceptStageData] = useState([]);
  const [filteredConceptStageData, setFilteredConceptStageData] = useState([]);
  const [dcNameFilter, setDcNameFilter] = useState('');
  const [stageFilter, setStageFilter] = useState('all');
  
  // Pagination state for concept stage table
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

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
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, []);

  // Apply filters when filters change
  useEffect(() => {
    let result = [...conceptStageData];
    
    // Apply dc_name filter
    if (dcNameFilter) {
      result = result.filter(item => 
        item.dc_name.toLowerCase().includes(dcNameFilter.toLowerCase())
      );
    }
    
    // Apply current_stage filter
    if (stageFilter !== 'all') {
      result = result.filter(item => item.current_stage === stageFilter);
    }
    
    setFilteredConceptStageData(result);
    // Reset to first page when filters change
    setPage(0);
  }, [dcNameFilter, stageFilter, conceptStageData]);

  // Get unique stages for dropdown
  const getUniqueStages = () => {
    const stages = [...new Set(conceptStageData.map(item => item.current_stage))];
    return stages;
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
                        label="概念板块名称"
                        variant="outlined"
                        fullWidth={false}
                        value={dcNameFilter}
                        onChange={(e) => setDcNameFilter(e.target.value)}
                        sx={{ minWidth: 250 }}
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
                            <TableCell>概念板块</TableCell>
                            <TableCell>提振</TableCell>
                            <TableCell>提振百分比</TableCell>
                            <TableCell>换手率</TableCell>
                            <TableCell>热度评分</TableCell>
                            <TableCell>当前阶段</TableCell>
                            <TableCell>阶段描述</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {filteredConceptStageData.length > 0 ? (
                            filteredConceptStageData
                              .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                              .map((item) => (
                                <TableRow key={item.key}>
                                  <TableCell><Typography component="a" href="#" color="primary">{item.dc_name}</Typography></TableCell>
                                  <TableCell>{item.uplift || 'N/A'}</TableCell>
                                  <TableCell>{safeFormatNumber(item.uplift_pct, 2, true)}</TableCell>
                                  <TableCell>{safeFormatNumber(item.turnover_ratio, 2, true)}</TableCell>
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
          </Grid>
          </Container>
        </section>
        <DefaultFooter />
      </div>
    </>
  );
};

export default TradePage;