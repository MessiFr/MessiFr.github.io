import React, { useState, useEffect } from 'react';
import { Card, CardBody, CardTitle, Container, Row, Col, Table, Input, UncontrolledDropdown, DropdownToggle, DropdownMenu, DropdownItem } from 'reactstrap';
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
  const [dcNameFilter, setDcNameFilter] = useState('');
  const [stageFilter, setStageFilter] = useState('all');
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });

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
          setConceptStageData(conceptResult.data);
          setFilteredConceptStageData(conceptResult.data);
        } else {
          console.error('Concept stage API request failed:', conceptResult.message);
        }
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, []);

  // Apply filters and sorting when filters change
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
    
    // Apply sorting
    if (sortConfig.key) {
      result.sort((a, b) => {
        let valA = a[sortConfig.key];
        let valB = b[sortConfig.key];
        
        // 如果是数字字符串，尝试转换为数字
        if(!isNaN(parseFloat(valA)) && !isNaN(parseFloat(valB))) {
          valA = parseFloat(valA);
          valB = parseFloat(valB);
        }
        
        if (valA < valB) {
          return sortConfig.direction === 'asc' ? -1 : 1;
        }
        if (valA > valB) {
          return sortConfig.direction === 'asc' ? 1 : -1;
        }
        return 0;
      });
    }
    
    setFilteredConceptStageData(result);
  }, [dcNameFilter, stageFilter, sortConfig, conceptStageData]);

  // Handle sorting
  const requestSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  // Get unique stages for dropdown
  const getUniqueStages = () => {
    const stages = [...new Set(conceptStageData.map(item => item.current_stage))];
    return stages;
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

  // Sortable header component
  const SortableHeader = ({ columnKey, columnName }) => {
    const isSorted = sortConfig.key === columnKey;
    const sortIndicator = isSorted 
      ? (sortConfig.direction === 'asc' ? ' ↑' : ' ↓') 
      : '';
    
    return (
      <th 
        onClick={() => requestSort(columnKey)}
        style={{ cursor: 'pointer' }}
      >
        {columnName}{sortIndicator}
      </th>
    );
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
          <Container>
            <Row className="justify-content-center">
              <Col lg="12">
                <h2 className="mb-4 text-center">交易日记</h2>
                
                {/* Summary Statistics in a single row */}
                {summaryStats && (
                  <Row className="mb-3">
                    <Col lg="12">
                      <div className="bg-light p-4 rounded text-dark">
                        <Row className="text-center">
                          <Col md="2" className="border-right">
                            <h5 className="text-uppercase text-dark ls-1 mb-1">初始本金</h5>
                            <span className="h4 font-weight-bold text-dark">{summaryStats.initialCapital}</span>
                          </Col>
                          <Col md="2" className="border-right">
                            <h5 className="text-uppercase text-dark ls-1 mb-1">最终总资产</h5>
                            <span className="h4 font-weight-bold text-dark">{summaryStats.finalAssets}</span>
                          </Col>
                          <Col md="2" className="border-right">
                            <h5 className="text-uppercase text-dark ls-1 mb-1">总收益率</h5>
                            <span className="h4 font-weight-bold text-dark">{summaryStats.totalReturn}</span>
                          </Col>
                          <Col md="3" className="border-right">
                            <h5 className="text-uppercase text-dark ls-1 mb-1">上证指数收益率</h5>
                            <span className="h4 font-weight-bold text-dark">{summaryStats.benchmarkReturn}</span>
                          </Col>
                          <Col md="3">
                            <h5 className="text-uppercase text-dark ls-1 mb-1">超额收益</h5>
                            <span className="h4 font-weight-bold text-dark">{summaryStats.excessReturn}</span>
                          </Col>
                        </Row>
                      </div>
                    </Col>
                  </Row>
                )}

                {/* Chart Section */}
                <Card className="shadow mb-5">
                  <CardBody>
                    <CardTitle tag="h3" className="mb-4 text-center">策略收益 vs 上证指数</CardTitle>
                    {chartData ? (
                      <div style={{ height: '550px' }}>
                        <Line data={chartData} options={options} />
                      </div>
                    ) : (
                      <p>Loading chart data...</p>
                    )}
                  </CardBody>
                </Card>

                {/* Trading History Table */}
                <Card className="shadow">
                  <CardBody>
                    <CardTitle tag="h3" className="mb-4 text-center">近期收益</CardTitle>
                    <Table responsive>
                      <thead>
                        <tr>
                          <th>Date</th>
                          <th>Total Assets</th>
                          <th>Position</th>
                          <th>Daily Return</th>
                          <th>Daily Return (%)</th>
                          <th>Cumulative Return</th>
                          <th>Holdings Count</th>
                        </tr>
                      </thead>
                      <tbody>
                        {tableData.map((row, index) => (
                          <tr key={index}>
                            <td>{row.date}</td>
                            <td>{row.totalAssets}</td>
                            <td>{row.portfolioValue}</td>
                            <td>{row.dailyReturn}</td>
                            <td>{row.dailyReturn1}</td>
                            <td>{row.cumReturn}</td>
                            <td>{row.holdings}</td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  </CardBody>
                </Card>

                {/* Concept Stage Table with Filters */}
                <Card className="shadow mt-5">
                  <CardBody>
                    <CardTitle tag="h3" className="mb-4 text-center">概念板块分析</CardTitle>
                    
                    {/* Filter Controls */}
                    <Row className="mb-3">
                      <Col md="6">
                        <label>概念板块名称:</label>
                        <Input
                          type="text"
                          placeholder="输入概念板块名称进行筛选..."
                          value={dcNameFilter}
                          onChange={(e) => setDcNameFilter(e.target.value)}
                        />
                      </Col>
                      <Col md="6">
                        <label>当前阶段:</label>
                        <UncontrolledDropdown block>
                          <DropdownToggle tag="div" className="btn-block text-left" caret>
                            {stageFilter === 'all' ? '全部阶段' : stageFilter}
                          </DropdownToggle>
                          <DropdownMenu container="body" className="w-100">
                            <DropdownItem 
                              onClick={() => setStageFilter('all')}
                              active={stageFilter === 'all'}
                            >
                              全部阶段
                            </DropdownItem>
                            {getUniqueStages().map((stage, index) => (
                              <DropdownItem 
                                key={index}
                                onClick={() => setStageFilter(stage)}
                                active={stageFilter === stage}
                              >
                                {stage}
                              </DropdownItem>
                            ))}
                          </DropdownMenu>
                        </UncontrolledDropdown>
                      </Col>
                    </Row>
                    
                    <Table responsive>
                      <thead>
                        <tr>
                          <SortableHeader columnKey="dc_name" columnName="概念板块" />
                          <SortableHeader columnKey="uplift" columnName="提振" />
                          <SortableHeader columnKey="uplift_pct" columnName="提振百分比" />
                          <SortableHeader columnKey="turnover_ratio" columnName="换手率" />
                          <SortableHeader columnKey="heat_score" columnName="热度评分" />
                          <SortableHeader columnKey="current_stage" columnName="当前阶段" />
                          <th>阶段描述</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredConceptStageData.length > 0 ? (
                          filteredConceptStageData.map((item, index) => (
                            <tr key={index}>
                              <td>{item.dc_name}</td>
                              <td>{item.uplift || 'N/A'}</td>
                              <td>{safeFormatNumber(item.uplift_pct, 2, true)}</td>
                              <td>{safeFormatNumber(item.turnover_ratio, 2, true)}</td>
                              <td>{safeFormatNumber(item.heat_score, 2)}</td>
                              <td>{item.current_stage}</td>
                              <td>{item.stage_desc}</td>
                            </tr>
                          ))
                        ) : (
                          <tr>
                            <td colSpan="7" className="text-center">没有找到匹配的数据</td>
                          </tr>
                        )}
                      </tbody>
                    </Table>
                  </CardBody>
                </Card>
              </Col>
            </Row>
          </Container>
        </section>
        <DefaultFooter />
      </div>
    </>
  );
};

export default TradePage;