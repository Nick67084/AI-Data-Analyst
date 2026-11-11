import {
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  PieChart,
  Pie,
  Cell,
} from "recharts";

import { api } from "./api";

import KpiCard from "./components/KpiCard";
import SectionCard from "./components/SectionCard";


function money(value) {
  return new Intl.NumberFormat(
    "en-US",
    {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0,
    }
  ).format(Number(value || 0));
}


function number(value) {
  return new Intl.NumberFormat(
    "en-US"
  ).format(Number(value || 0));
}


function revenueFromForecast(row) {
  return Number(
    row.PredictedRevenue ??
      row.Predicted_Sales ??
      row.Revenue ??
      row.revenue ??
      0
  );
}


function revenueFromAnomaly(row) {
  return Number(
    row.Revenue ??
      row.revenue ??
      0
  );
}


export default function App() {

  const [data, setData] =
    useState({
      kpis: null,
      monthlyRevenue: [],
      topProducts: [],
      forecast: [],
      anomalies: [],
      segments: [],
      mlStatus: null,
    });


  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");


  // ----------------------------------------------
  // AI Analyst state
  // ----------------------------------------------

  const [ragStatus, setRagStatus] =
    useState(null);

  const [selectedFile, setSelectedFile] =
    useState(null);

  const [uploading, setUploading] =
    useState(false);

  const [uploadMessage, setUploadMessage] =
    useState("");

  const [question, setQuestion] =
    useState("");

  const [asking, setAsking] =
    useState(false);

  const [aiAnswer, setAiAnswer] =
    useState("");

  const [sources, setSources] =
    useState([]);

  const [analysisContext, setAnalysisContext] =
    useState(null);

  const [ragError, setRagError] =
    useState("");


  // ----------------------------------------------
  // Load dashboard
  // ----------------------------------------------

  async function loadDashboard() {

    setLoading(true);
    setError("");

    try {

      const [
        kpis,
        monthlyRevenue,
        topProducts,
        forecast,
        anomalies,
        segments,
        mlStatus,
        rag,
      ] = await Promise.all([

        api.getKpis(),

        api.getMonthlyRevenue(),

        api.getTopProducts(),

        api.getForecast(),

        api.getAnomalies(),

        api.getCustomerSegments(),

        api.getMlStatus(),

        api.getRagStatus(),

      ]);


      setData({
        kpis,
        monthlyRevenue,
        topProducts,
        forecast,
        anomalies,
        segments,
        mlStatus,
      });


      setRagStatus(rag);

    } catch (err) {

      setError(
        "Could not connect to the FastAPI backend. " +
        "Make sure Uvicorn is running on port 8000."
      );

      console.error(err);

    } finally {

      setLoading(false);

    }
  }


  useEffect(() => {
    loadDashboard();
  }, []);


  // ----------------------------------------------
  // Chart data
  // ----------------------------------------------

  const segmentChartData =
    useMemo(
      () =>
        data.segments.map(
          (item) => ({
            name: item.Segment,
            customers:
              Number(
                item.customers || 0
              ),
          })
        ),

      [data.segments]
    );


  const forecastChartData =
    useMemo(
      () =>
        data.forecast.map(
          (row) => ({
            date:
              row.Date ??
              row.date,

            revenue:
              revenueFromForecast(row),
          })
        ),

      [data.forecast]
    );


  // ----------------------------------------------
  // Upload document
  // ----------------------------------------------

  async function handleUpload() {

    if (!selectedFile) {

      setRagError(
        "Please choose a PDF, TXT, or DOCX file first."
      );

      return;
    }


    setUploading(true);

    setRagError("");

    setUploadMessage("");

    setAiAnswer("");

    setSources([]);

    setAnalysisContext(null);


    try {

      const result =
        await api.uploadDocument(
          selectedFile
        );


      setUploadMessage(
        `${result.filename} processed successfully — ` +
        `${result.chunks} text chunk${
          result.chunks === 1
            ? ""
            : "s"
        } created.`
      );


      const status =
        await api.getRagStatus();

      setRagStatus(status);


    } catch (err) {

      setRagError(
        err.message ||
        "Document upload failed."
      );

    } finally {

      setUploading(false);

    }
  }


  // ----------------------------------------------
  // Ask combined AI Analyst
  // ----------------------------------------------

  async function handleAsk() {

    const trimmedQuestion =
      question.trim();


    if (!trimmedQuestion) {

      setRagError(
        "Please enter a question."
      );

      return;
    }


    if (
      !ragStatus?.vector_store_ready
    ) {

      setRagError(
        "Please upload and process a business document first."
      );

      return;
    }


    setAsking(true);

    setRagError("");

    setAiAnswer("");

    setSources([]);

    setAnalysisContext(null);


    try {

      const result =
        await api.askAnalyst(
          trimmedQuestion,
          5
        );


      setAiAnswer(
        result.answer ||
        "No answer was returned."
      );


      setSources(
        result.document_sources ||
        []
      );


      setAnalysisContext(
        result.data_context ||
        null
      );


    } catch (err) {

      setRagError(
        err.message ||
        "The AI could not answer the question."
      );

    } finally {

      setAsking(false);

    }
  }


  return (
    <div className="app">

      {/* =========================================
          HEADER
      ========================================= */}

      <header className="topbar">

        <div>

          <div className="eyebrow">
            AI-POWERED BUSINESS INTELLIGENCE
          </div>

          <h1>
            AI Data Analyst
          </h1>

          <p>
            Retail analytics, machine learning,
            forecasting, customer intelligence,
            and document-based AI in one dashboard.
          </p>

        </div>


        <button
          className="refresh-button"
          onClick={loadDashboard}
        >
          Refresh Data
        </button>

      </header>


      <main className="dashboard">

        {/* =========================================
            LOADING / ERROR
        ========================================= */}

        {loading && (
          <div className="status-card">
            Loading dashboard...
          </div>
        )}


        {error && (
          <div className="error-card">

            <strong>
              Backend connection error
            </strong>

            <span>
              {error}
            </span>

          </div>
        )}


        {!loading &&
          !error &&
          data.kpis && (

          <>

            {/* =====================================
                AI ANALYST
            ===================================== */}

            <section className="ai-section">

              <div className="ai-section-header">

                <div>

                  <div className="ai-label">
                    GENERATIVE AI + RAG
                  </div>

                  <h2>
                    AI Analyst
                  </h2>

                  <p>
                    Ask questions about your sales
                    data and uploaded business documents.
                    The analyst combines analytics,
                    machine learning results,
                    retrieved document context,
                    and Llama 3.2.
                  </p>

                </div>


                <div className="rag-status-pill">

                  <span
                    className={
                      ragStatus?.vector_store_ready
                        ? "status-dot online"
                        : "status-dot"
                    }
                  />

                  {ragStatus?.vector_store_ready
                    ? `${number(
                        ragStatus.stored_chunks
                      )} chunks ready`
                    : "No document loaded"}

                </div>

              </div>


              <div className="ai-grid">

                {/* DOCUMENT UPLOAD */}

                <div className="ai-panel">

                  <label
                    className="field-label"
                    htmlFor="document-upload"
                  >
                    Business document
                  </label>


                  <input
                    id="document-upload"
                    className="file-input"
                    type="file"
                    accept=".pdf,.txt,.docx"
                    onChange={(event) => {

                      setSelectedFile(
                        event.target.files?.[0] ||
                        null
                      );

                      setUploadMessage("");

                      setRagError("");

                    }}
                  />


                  {selectedFile && (

                    <div className="selected-file">

                      Selected:

                      {" "}

                      <strong>
                        {selectedFile.name}
                      </strong>

                    </div>

                  )}


                  <button
                    className="primary-button"
                    onClick={handleUpload}
                    disabled={uploading}
                  >

                    {uploading
                      ? "Processing document..."
                      : "Upload & Process"}

                  </button>


                  {uploadMessage && (

                    <div className="success-message">
                      {uploadMessage}
                    </div>

                  )}

                </div>


                {/* QUESTION */}

                <div className="ai-panel">

                  <label
                    className="field-label"
                    htmlFor="ai-question"
                  >
                    Ask your business question
                  </label>


                  <textarea
                    id="ai-question"
                    className="question-input"
                    rows="5"
                    placeholder={
                      "Example: Why did revenue increase during Q4, " +
                      "and what marketing strategy contributed to it?"
                    }
                    value={question}
                    onChange={(event) =>
                      setQuestion(
                        event.target.value
                      )
                    }
                  />


                  <button
                    className="primary-button"
                    onClick={handleAsk}
                    disabled={asking}
                  >

                    {asking
                      ? "Analyzing business data..."
                      : "Ask AI Analyst"}

                  </button>

                </div>

              </div>


              {/* ERROR */}

              {ragError && (

                <div className="rag-error">
                  {ragError}
                </div>

              )}


              {/* =================================
                  AI ANSWER
              ================================= */}

              {aiAnswer && (

                <div className="answer-panel">

                  <div className="answer-heading">
                    AI Analyst Answer
                  </div>


                  <div className="answer-text">
                    {aiAnswer}
                  </div>


                  {/* DATA USED */}

                  {analysisContext && (

                    <div className="sources-section">

                      <div className="source-heading">
                        Structured Data Used
                      </div>


                      <div className="data-used-grid">

                        <span>
                          Revenue:{" "}
                          {money(
                            analysisContext
                              .kpis
                              ?.total_revenue
                          )}
                        </span>


                        <span>
                          Orders:{" "}
                          {number(
                            analysisContext
                              .kpis
                              ?.total_orders
                          )}
                        </span>


                        <span>
                          Customers:{" "}
                          {number(
                            analysisContext
                              .kpis
                              ?.total_customers
                          )}
                        </span>


                        <span>
                          Forecast days:{" "}
                          {number(
                            analysisContext
                              .forecast_days
                          )}
                        </span>


                        <span>
                          Anomalies reviewed:{" "}
                          {number(
                            analysisContext
                              .anomalies_count
                          )}
                        </span>


                        <span>
                          Customer segments:{" "}
                          {number(
                            analysisContext
                              .customer_segments_count
                          )}
                        </span>

                      </div>

                    </div>

                  )}


                  {/* DOCUMENT SOURCES */}

                  {sources.length > 0 && (

                    <div className="sources-section">

                      <div className="source-heading">
                        Retrieved Business Sources
                      </div>


                      {sources.map(
                        (
                          source,
                          index
                        ) => (

                          <details
                            className="source-item"
                            key={
                              `${source.source}-${index}`
                            }
                          >

                            <summary>
                              {source.source}
                              {" · "}
                              similarity{" "}
                              {source.score}
                            </summary>


                            <p>
                              {source.text}
                            </p>

                          </details>

                        )
                      )}

                    </div>

                  )}

                </div>

              )}

            </section>


            {/* =====================================
                KPI CARDS
            ===================================== */}

            <div className="kpi-grid">

              <KpiCard
                title="Total Revenue"
                value={money(
                  data.kpis.total_revenue
                )}
                subtitle="All cleaned transactions"
              />


              <KpiCard
                title="Total Orders"
                value={number(
                  data.kpis.total_orders
                )}
                subtitle="Unique invoices"
              />


              <KpiCard
                title="Customers"
                value={number(
                  data.kpis.total_customers
                )}
                subtitle="Identified customers"
              />


              <KpiCard
                title="Average Order Value"
                value={money(
                  data.kpis.average_order_value
                )}
                subtitle="Average revenue per order"
              />

            </div>


            {/* =====================================
                MONTHLY REVENUE + SEGMENTS
            ===================================== */}

            <div className="two-column">

              <SectionCard title="Monthly Revenue">

                <div className="chart">

                  <ResponsiveContainer
                    width="100%"
                    height={330}
                  >

                    <LineChart
                      data={
                        data.monthlyRevenue
                      }
                    >

                      <CartesianGrid
                        strokeDasharray="3 3"
                      />

                      <XAxis
                        dataKey="month"
                      />

                      <YAxis
                        tickFormatter={(value) =>
                          `${Math.round(
                            value / 1000
                          )}k`
                        }
                      />

                      <Tooltip
                        formatter={(value) =>
                          money(value)
                        }
                      />

                      <Line
                        type="monotone"
                        dataKey="revenue"
                        stroke="#2563eb"
                        strokeWidth={3}
                        dot={false}
                      />

                    </LineChart>

                  </ResponsiveContainer>

                </div>

              </SectionCard>


              <SectionCard title="Customer Segments">

                <div className="chart">

                  <ResponsiveContainer
                    width="100%"
                    height={330}
                  >

                    <PieChart>

                      <Pie
                        data={
                          segmentChartData
                        }
                        dataKey="customers"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        outerRadius={105}
                        label
                      >

                        {segmentChartData.map(
                          (_, index) => (

                            <Cell
                              key={index}
                              fill={
                                [
                                  "#2563eb",
                                  "#64748b",
                                  "#16a34a",
                                  "#f59e0b",
                                ][
                                  index % 4
                                ]
                              }
                            />

                          )
                        )}

                      </Pie>

                      <Tooltip />

                      <Legend />

                    </PieChart>

                  </ResponsiveContainer>

                </div>

              </SectionCard>

            </div>


            {/* =====================================
                PRODUCTS + FORECAST
            ===================================== */}

            <div className="two-column">

              <SectionCard title="Top Products by Revenue">

                <div className="chart">

                  <ResponsiveContainer
                    width="100%"
                    height={360}
                  >

                    <BarChart
                      data={
                        data.topProducts
                      }
                      layout="vertical"
                      margin={{
                        left: 20,
                        right: 25,
                      }}
                    >

                      <CartesianGrid
                        strokeDasharray="3 3"
                      />

                      <XAxis
                        type="number"
                        tickFormatter={(value) =>
                          `${Math.round(
                            value / 1000
                          )}k`
                        }
                      />

                      <YAxis
                        type="category"
                        dataKey="Description"
                        width={180}
                        tick={{
                          fontSize: 11,
                        }}
                      />

                      <Tooltip
                        formatter={(value) =>
                          money(value)
                        }
                      />

                      <Bar
                        dataKey="revenue"
                        fill="#2563eb"
                      />

                    </BarChart>

                  </ResponsiveContainer>

                </div>

              </SectionCard>


              <SectionCard title="Revenue Forecast">

                <div className="chart">

                  <ResponsiveContainer
                    width="100%"
                    height={360}
                  >

                    <LineChart
                      data={
                        forecastChartData
                      }
                    >

                      <CartesianGrid
                        strokeDasharray="3 3"
                      />

                      <XAxis
                        dataKey="date"
                      />

                      <YAxis
                        tickFormatter={(value) =>
                          `${Math.round(
                            value / 1000
                          )}k`
                        }
                      />

                      <Tooltip
                        formatter={(value) =>
                          money(value)
                        }
                      />

                      <Line
                        type="monotone"
                        dataKey="revenue"
                        stroke="#16a34a"
                        strokeWidth={3}
                        dot={false}
                      />

                    </LineChart>

                  </ResponsiveContainer>

                </div>

              </SectionCard>

            </div>


            {/* =====================================
                ANOMALIES + SEGMENT TABLE
            ===================================== */}

            <div className="two-column">

              <SectionCard
                title={
                  `Detected Anomalies ` +
                  `(${data.anomalies.length})`
                }
              >

                <div className="table-wrapper">

                  <table>

                    <thead>

                      <tr>

                        <th>
                          Date
                        </th>

                        <th>
                          Revenue
                        </th>

                      </tr>

                    </thead>


                    <tbody>

                      {data.anomalies
                        .slice(0, 10)
                        .map(
                          (
                            row,
                            index
                          ) => (

                            <tr
                              key={
                                `${row.Date}-${index}`
                              }
                            >

                              <td>
                                {row.Date}
                              </td>

                              <td>
                                {money(
                                  revenueFromAnomaly(
                                    row
                                  )
                                )}
                              </td>

                            </tr>

                          )
                        )}

                    </tbody>

                  </table>

                </div>

              </SectionCard>


              <SectionCard title="Customer Segment Summary">

                <div className="table-wrapper">

                  <table>

                    <thead>

                      <tr>

                        <th>
                          Segment
                        </th>

                        <th>
                          Customers
                        </th>

                        <th>
                          Avg. Monetary
                        </th>

                      </tr>

                    </thead>


                    <tbody>

                      {data.segments.map(
                        (row) => (

                          <tr
                            key={
                              row.Segment
                            }
                          >

                            <td>
                              {row.Segment}
                            </td>

                            <td>
                              {number(
                                row.customers
                              )}
                            </td>

                            <td>
                              {money(
                                row.avg_monetary
                              )}
                            </td>

                          </tr>

                        )
                      )}

                    </tbody>

                  </table>

                </div>

              </SectionCard>

            </div>


            {/* =====================================
                ML STATUS
            ===================================== */}

            <SectionCard title="ML Model Status">

              <div className="model-status">

                {Object.entries(
                  data.mlStatus || {}
                ).map(
                  (
                    [
                      name,
                      status,
                    ]
                  ) => (

                    <div
                      className="model-item"
                      key={name}
                    >

                      <span>
                        {name.replaceAll(
                          "_",
                          " "
                        )}
                      </span>


                      <span
                        className={
                          status
                            ? "badge success"
                            : "badge danger"
                        }
                      >
                        {status
                          ? "Available"
                          : "Missing"}
                      </span>

                    </div>

                  )
                )}

              </div>

            </SectionCard>

          </>
        )}

      </main>


      <footer>
        AI Data Analyst · React · FastAPI · Pandas ·
        Machine Learning · RAG · Ollama
      </footer>

    </div>
  );
}