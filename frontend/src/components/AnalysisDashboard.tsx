import { useEffect, useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import axios from 'axios';

interface AnalysisDashboardProps {
  sessionId: string;
}

interface AnalysisData {
  summary: {
    total_columns: number;
    total_rows: number;
    column_types: Record<string, string>;
    missing_values: Record<string, number>;
  };
  column_analysis: Record<string, any>;
  correlations: Record<string, number>;
  insights: Array<{
    type: string;
    column: string;
    message: string;
    details: Record<string, any>;
  }>;
  data_quality: {
    completeness: Record<string, number>;
    unique_ratio: Record<string, number>;
  };
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

const AnalysisDashboard = ({ sessionId }: AnalysisDashboardProps) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);

  useEffect(() => {
    const fetchAnalysis = async () => {
      try {
        const response = await axios.post(
          `http://localhost:8000/api/v1/analyze/${sessionId}`
        );
        setAnalysisData(response.data);
      } catch (err) {
        setError('Failed to fetch analysis data');
      } finally {
        setLoading(false);
      }
    };

    fetchAnalysis();
  }, [sessionId]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" p={4}>
        <CircularProgress />
      </Box>
    );
  }

  if (error || !analysisData) {
    return (
      <Alert severity="error" sx={{ width: '100%' }}>
        {error || 'Failed to load analysis'}
      </Alert>
    );
  }

  const dataQualityData = Object.entries(analysisData.data_quality.completeness).map(
    ([column, value]) => ({
      name: column,
      completeness: value,
      uniqueness: analysisData.data_quality.unique_ratio[column] || 0,
    })
  );

  const insightsByType = analysisData.insights.reduce((acc, insight) => {
    if (!acc[insight.type]) {
      acc[insight.type] = [];
    }
    acc[insight.type].push(insight);
    return acc;
  }, {} as Record<string, typeof analysisData.insights>);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      <Typography variant="h5" gutterBottom>
        Data Analysis Dashboard
      </Typography>

      <Grid container spacing={3}>
        {/* Summary Statistics */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Dataset Summary
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6} md={3}>
                <Typography variant="body2" color="text.secondary">
                  Total Rows
                </Typography>
                <Typography variant="h6">
                  {analysisData.summary.total_rows}
                </Typography>
              </Grid>
              <Grid item xs={6} md={3}>
                <Typography variant="body2" color="text.secondary">
                  Total Columns
                </Typography>
                <Typography variant="h6">
                  {analysisData.summary.total_columns}
                </Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Data Quality Chart */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Data Quality Metrics
            </Typography>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={dataQualityData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="completeness" fill="#8884d8" name="Completeness %" />
                <Bar dataKey="uniqueness" fill="#82ca9d" name="Uniqueness %" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Insights */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Key Insights
            </Typography>
            {Object.entries(insightsByType).map(([type, insights]) => (
              <Box key={type} sx={{ mb: 2 }}>
                <Typography variant="subtitle1" sx={{ mb: 1 }}>
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </Typography>
                {insights.map((insight, index) => (
                  <Alert
                    key={index}
                    severity="info"
                    sx={{ mb: 1 }}
                  >
                    {insight.message}
                  </Alert>
                ))}
              </Box>
            ))}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default AnalysisDashboard; 