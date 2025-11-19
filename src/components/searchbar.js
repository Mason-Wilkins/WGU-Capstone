import React, { useState, useEffect } from 'react';
import axios from 'axios';

function SearchBar() {
    const [message, setmessage] = useState('');
    useEffect(() => {
        fetch('http://localhost:5000/api/data')
            .then(response => response.json())
            .then(data => console.log(data))
            .catch(error => console.error('Error fetching data:', error));
        // axios.get('http://localhost:3000/api/data')
        //     .then(response => {
        //         console.log(response)
        //         setmessage(response.data.message);
        //     }
        //     )
        //     .catch(error => {
        //         console.error('There was an error fetching the data!', error);
        //     });
    }
        , []);
    return (
        <div className="SearchBar">
            <input
                type="text"
                placeholder="Search..."
            />
            <p>
                {message}
                "Hello, world!"
            </p>
        </div>
    );
}

export default SearchBar; 