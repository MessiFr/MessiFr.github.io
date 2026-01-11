import React, { useState, useEffect } from 'react';
import { Card, CardBody, CardTitle, Container, Row, Col, Table } from 'reactstrap';
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

  useEffect(() => {
    // Fetch real data from API
    const fetchData = async () => {
      try {
        const response = await fetch('/api/portfolio-daily');
        const result = await response.json();

        if (result.success) {
          const data = result.data;

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
          console.error('API request failed:', result.message);
        }
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, []);

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