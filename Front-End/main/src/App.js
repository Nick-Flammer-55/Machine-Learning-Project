import React, {useEffect, useState} from 'react';
import { Typography, Container, Slider, Box } from '@mui/material';
import TeamPicker from './components/TeamPicker';
import axios from 'axios';
import teamAssets from './components/team_info.json';
import './App.css';

function App() {
  const [homeTeam, setHomeTeam] = useState(null);
  const [guestTeam, setGuestTeam] = useState(null);
  //const [homeTeamData, setHomeTeamData] = useState(null);
  //const [guestTeamData, setGuestTeamData] = useState(null);
  const [winningTeam, setWinningTeam] = useState("");
  const [winningTeamPercentage, setWinningTeamPercentage] = useState(0);

  // useEffect(() => {
  //   // setHomeTeamData(teamAssets.teams.find(item => item.team === homeTeam));
  //   // setGuestTeamData(teamAssets.teams.find(item => item.team === guestTeam));
  // }, [homeTeam, guestTeam])

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
        <Container sx={{
          height: '25vh',
          width: '75vh',
          display: 'flex',
          justifyContent: 'space-evenly',
          alignItems: 'center',
          padding: '10px',
        }}>
          {(homeTeam && guestTeam) && <Slider sx={{
            height: 10,
            padding: '15px 0',
            color: 'transparent',
    
            '& .MuiSlider-thumb': {
              display: 'none',
            },
    
            '& .MuiSlider-track': {
              backgroundColor: homeTeam.primary_color,
              border: '2px solid ' + homeTeam.secondary_color,
              borderRadius: 2,
            },
    
            '& .MuiSlider-rail': {
              backgroundColor: guestTeam.primary_color,
              border: '2px solid ' + guestTeam.secondary_color,
              borderRadius: 2,
              opacity: 1,
            },

          }}
          defaultValue={50}
          />}  
        </Container>
    </div>
  );
}

export default App;
