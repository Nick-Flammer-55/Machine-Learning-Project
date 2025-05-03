import React, {useEffect, useState} from 'react';
import { Typography, Container, Slider, Box } from '@mui/material';
import TeamPicker from './components/TeamPicker';
import axios from 'axios';
import './App.css';

function App() {
  const [homeTeam, setHomeTeam] = useState(null);
  const [guestTeam, setGuestTeam] = useState(null);
  const [winningTeam, setWinningTeam] = useState("");
  const [winningTeamPercentage, setWinningTeamPercentage] = useState(0);

  useEffect(() => {
    const handlePrediction = async () => {
      try {
        const res = await axios.post(`http://127.0.0.1/predict`, { homeTeam, guestTeam });
        setWinningTeam(res.data.winner);
        setWinningTeamPercentage(res.data.winner_probability)
      } catch (error) {
        console.error("Error fetching prediction result:", error);
      }
    };
    handlePrediction()
  }, [homeTeam, guestTeam])
    // const handlePrediction = async () => {
    //   try {
    //     const res = await axios.post(`${process.env.REACT_APP_API_URL}/predict`, { homeTeam, guestTeam });
    //     setWinningTeam(res.data.winner);
    //     setWinningTeamPercentage(res.data.winner_probability)
    //   } catch (error) {
    //     console.error("Error fetching prediction result:", error);
    //   }
    // }

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
