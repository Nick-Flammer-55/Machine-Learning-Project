import React, {useState} from 'react';
import { Typography, Container } from '@mui/material';
import TeamPicker from './components/TeamPicker';
import './App.css';

function App() {
  const [homeTeam, setHomeTeam] = useState("")
  const [guestTeam, setGuestTeam] = useState("")
  
  return (
    <div className='App'>
      <Container
        sx={{
          height: '15vh',
          display: 'flex',
          justifyContent: 'space-evenly',
          alignItems: 'center',
          padding: '10px',
        }}  
      >
        <img src="/assets/logo-nba.svg" alt="" width={150}/>
      </Container>
      <Container
        sx={{
          height: '25vh',
          display: 'flex',
          justifyContent: 'space-evenly',
          alignItems: 'center',
          padding: '10px',
        }}  
      >
        <TeamPicker 
          team={homeTeam}
          opTeam={guestTeam}
          setTeam = {setHomeTeam}
        />
        <Typography>VS</Typography>
        <TeamPicker 
          team={guestTeam}
          opTeam={homeTeam}
          setTeam={setGuestTeam}
        />
      </Container>
    </div>
  );
}

export default App;
