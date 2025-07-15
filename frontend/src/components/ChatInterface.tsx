import { useState } from 'react';
import {
  Box,
  Paper,
  TextField,
  IconButton,
  Typography,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import { Send as SendIcon } from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts';
import axios from 'axios';

interface ChatInterfaceProps {
  sessionId: string;
}

interface ChatMessage {
  type: 'user' | 'assistant';
  content: string;
  visualization?: {
    type: string;
    data: any;
  };
  statistics?: Record<string, any>;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

const ChatInterface = ({ sessionId }: ChatInterfaceProps) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = input.trim();
    setInput('');
    setLoading(true);
    setError(null);

    setMessages((prev) => [...prev, { type: 'user', content: userMessage }]);

    try {
      const response = await axios.post(
        `http://localhost:8000/api/v1/chat/${sessionId}`,
        { query: userMessage }
      );

      const assistantMessage: ChatMessage = {
        type: 'assistant',
        content: response.data.answer,
      };

      if (response.data.visualization_data) {
        assistantMessage.visualization = {
          type: response.data.visualization_data.type,
          data: response.data.visualization_data.data,
        };
      }

      if (response.data.statistics) {
        assistantMessage.statistics = response.data.statistics;
      }

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to get response';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const renderStatistics = (statistics: Record<string, any>) => {
    return (
      <TableContainer component={Paper} sx={{ mt: 2 }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Column</TableCell>
              <TableCell>Metric</TableCell>
              <TableCell align="right">Value</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {Object.entries(statistics).map(([column, stats]) => (
              Object.entries(stats).map(([metric, value]) => (
                <TableRow key={`${column}-${metric}`}>
                  <TableCell>{column}</TableCell>
                  <TableCell>{metric}</TableCell>
                  <TableCell align="right">
                    {typeof value === 'number' ? value.toFixed(2) : String(value)}
                  </TableCell>
                </TableRow>
              ))
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };

  const renderVisualization = (visualization: ChatMessage['visualization']) => {
    if (!visualization) return null;

    const { type, data } = visualization;
    const height = 300;

    switch (type) {
      case 'line':
        return (
          <Box sx={{ height, width: '100%', mt: 2 }}>
            <ResponsiveContainer>
              <LineChart data={Object.values(data)[0].values}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis />
                <YAxis />
                <Tooltip />
                <Legend />
                {Object.keys(data).map((key, index) => (
                  <Line
                    key={key}
                    type="monotone"
                    dataKey="value"
                    name={key}
                    stroke={COLORS[index % COLORS.length]}
                    dot={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </Box>
        );

      case 'bar':
        return (
          <Box sx={{ height, width: '100%', mt: 2 }}>
            <ResponsiveContainer>
              <BarChart data={Object.values(data)[0].labels.map((label: string, index: number) => ({
                name: label,
                value: Object.values(data)[0].values[index]
              }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" fill={COLORS[0]} />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        );

      case 'pie':
        return (
          <Box sx={{ height, width: '100%', mt: 2 }}>
            <ResponsiveContainer>
              <PieChart>
                <Pie
                  data={Object.values(data)[0].labels.map((label: string, index: number) => ({
                    name: label,
                    value: Object.values(data)[0].values[index]
                  }))}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  label
                >
                  {Object.values(data)[0].labels.map((_: any, index: number) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Box>
        );

      case 'comparison':
        return (
          <Box sx={{ height, width: '100%', mt: 2 }}>
            <ResponsiveContainer>
              <BarChart data={Object.entries(data).flatMap(([column, columnData]: [string, any]) =>
                columnData.labels.map((label: string, index: number) => ({
                  name: label,
                  [column]: columnData.values[index]
                }))
              )}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                {Object.keys(data).map((key, index) => (
                  <Bar key={key} dataKey={key} fill={COLORS[index % COLORS.length]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Data Analysis Chat
      </Typography>

      <Box
        sx={{
          height: 400,
          mb: 2,
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
        }}
      >
        {messages.map((message, index) => (
          <Box
            key={index}
            sx={{
              alignSelf: message.type === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: '80%',
              width: message.visualization ? '100%' : 'auto',
            }}
          >
            <Paper
              sx={{
                p: 2,
                bgcolor: message.type === 'user' ? 'primary.main' : 'grey.100',
                color: message.type === 'user' ? 'white' : 'text.primary',
              }}
            >
              <Typography>{message.content}</Typography>
            </Paper>
            {message.visualization && renderVisualization(message.visualization)}
            {message.statistics && renderStatistics(message.statistics)}
          </Box>
        ))}
        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center' }}>
            <CircularProgress size={24} />
          </Box>
        )}
        {error && (
          <Alert severity="error" sx={{ width: '100%' }}>
            {error}
          </Alert>
        )}
      </Box>

      <Box sx={{ display: 'flex', gap: 1 }}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Ask about your data (e.g., 'Show order status distribution', 'Compare sales by region')"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
        />
        <IconButton
          color="primary"
          onClick={handleSend}
          disabled={loading || !input.trim()}
        >
          <SendIcon />
        </IconButton>
      </Box>
    </Paper>
  );
};

export default ChatInterface; 