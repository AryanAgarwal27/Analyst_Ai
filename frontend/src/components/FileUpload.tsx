import { useState } from 'react';
import { 
  Box, 
  Button, 
  Typography, 
  Paper,
  CircularProgress,
  Alert
} from '@mui/material';
import { CloudUpload } from '@mui/icons-material';
import axios from 'axios';

interface FileUploadProps {
  onUploadSuccess: (sessionId: string) => void;
}

const FileUpload = ({ onUploadSuccess }: FileUploadProps) => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const initializeSession = async () => {
    try {
      const response = await axios.post('http://localhost:8000/api/v1/init');
      return response.data.session_id;
    } catch (err) {
      console.error('Session initialization error:', err);
      throw new Error('Failed to initialize session');
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const selectedFile = event.target.files[0];
      console.log('Selected file:', selectedFile.name, 'Type:', selectedFile.type);
      
      // Accept more MIME types and file extensions
      const validTypes = [
        'text/csv',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/csv',
        'text/x-csv',
        'application/x-csv',
      ];
      
      const validExtensions = ['.csv', '.xls', '.xlsx'];
      const fileExtension = selectedFile.name.toLowerCase().slice(selectedFile.name.lastIndexOf('.'));
      
      if (validTypes.includes(selectedFile.type) || validExtensions.includes(fileExtension)) {
        setFile(selectedFile);
        setError(null);
      } else {
        setError(`Invalid file type. Please upload a CSV or Excel file. Received type: ${selectedFile.type}`);
        setFile(null);
      }
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const sessionId = await initializeSession();
      console.log('Session initialized:', sessionId);
      
      const formData = new FormData();
      formData.append('file', file);

      console.log('Uploading file:', file.name);
      const response = await axios.post(
        `http://localhost:8000/api/v1/upload/${sessionId}`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      console.log('Upload response:', response.data);
      onUploadSuccess(sessionId);
    } catch (err: any) {
      console.error('Upload error:', err);
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to upload file';
      setError(`Error: ${errorMessage}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Paper 
      elevation={3}
      sx={{
        p: 4,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 3
      }}
    >
      <Typography variant="h4" component="h1" gutterBottom>
        Analyst AI
      </Typography>
      
      <Typography variant="body1" color="text.secondary" align="center">
        Upload your CSV or Excel file to start analyzing your data
      </Typography>

      <Box
        sx={{
          width: '100%',
          maxWidth: 500,
          display: 'flex',
          flexDirection: 'column',
          gap: 2
        }}
      >
        <Button
          component="label"
          variant="outlined"
          startIcon={<CloudUpload />}
          sx={{ height: 56 }}
          fullWidth
        >
          Choose File
          <input
            type="file"
            hidden
            accept=".csv,.xlsx,.xls"
            onChange={handleFileChange}
          />
        </Button>

        {file && (
          <Typography variant="body2" color="text.secondary" align="center">
            Selected file: {file.name} ({file.type || 'unknown type'})
          </Typography>
        )}

        {error && (
          <Alert severity="error" sx={{ width: '100%' }}>
            {error}
          </Alert>
        )}

        <Button
          variant="contained"
          onClick={handleUpload}
          disabled={!file || loading}
          sx={{ height: 56 }}
          fullWidth
        >
          {loading ? (
            <CircularProgress size={24} color="inherit" />
          ) : (
            'Upload and Analyze'
          )}
        </Button>
      </Box>
    </Paper>
  );
};

export default FileUpload; 