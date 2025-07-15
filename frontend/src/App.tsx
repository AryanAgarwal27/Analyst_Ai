import { useState } from 'react';
import { ThemeProvider, CssBaseline, Box, Container } from '@mui/material';
import { createTheme } from '@mui/material/styles';
import FileUpload from './components/FileUpload';
import AnalysisDashboard from './components/AnalysisDashboard';
import ChatInterface from './components/ChatInterface';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [dataLoaded, setDataLoaded] = useState(false);

  const handleFileUploadSuccess = (newSessionId: string) => {
    setSessionId(newSessionId);
    setDataLoaded(true);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ 
        minHeight: '100vh',
        bgcolor: 'background.default',
        py: 4
      }}>
        <Container maxWidth="lg">
          {!dataLoaded ? (
            <FileUpload onUploadSuccess={handleFileUploadSuccess} />
          ) : (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <AnalysisDashboard sessionId={sessionId!} />
              <ChatInterface sessionId={sessionId!} />
            </Box>
          )}
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App; 