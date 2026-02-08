'use client';

import React, { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js').then(mod => mod.default), { 
    ssr: false,
    loading: () => (
        <div className="w-full h-full flex items-center justify-center bg-gray-900 animate-pulse rounded-3xl">
            <span className="text-gray-500">Loading Map...</span>
        </div>
    )
});

interface MapData {
    FIPS: string;
    State: string;
    County: string;
    median_aqi: number;
    population_density: number;
    Median_Household_Income: number;
}

export default function AQIChoropleth() {
    const [data, setData] = useState<MapData[] | null>(null);
    const [geojson, setGeojson] = useState<any>(null);

    useEffect(() => {
        // Fetch processed AQI data
        fetch('/data/aqi_map_data.json')
            .then(res => {
                if (!res.ok) throw new Error('Failed to load AQI data');
                return res.json();
            })
            .then(setData)
            .catch(err => console.error('AQI Data Error:', err));

        // Fetch USA Counties GeoJSON
        fetch('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json')
            .then(res => {
                if (!res.ok) throw new Error('Failed to load GeoJSON');
                return res.json();
            })
            .then(setGeojson)
            .catch(err => console.error('GeoJSON Error:', err));
    }, []);

    if (!data || !geojson) {
        return (
            <div className="w-full h-full flex items-center justify-center bg-gray-900 rounded-3xl">
                <div className="text-center">
                    <div className="inline-block w-8 h-8 border-4 border-accent-start border-t-transparent rounded-full animate-spin mb-4"></div>
                    <p className="text-gray-400">Initializing Geographic Data...</p>
                </div>
            </div>
        );
    }

    const locations = data.map(d => d.FIPS);
    const z = data.map(d => d.median_aqi);
    const text = data.map(d => `${d.County}, ${d.State}<br>Median AQI: ${d.median_aqi}<br>Income: $${d.Median_Household_Income.toLocaleString()}`);

    return (
        <div className="w-full h-full rounded-3xl overflow-hidden bg-gray-900">
            <Plot
                data={[
                    {
                        type: 'choropleth',
                        geojson: geojson,
                        locations: locations,
                        z: z,
                        text: text,
                        hoverinfo: 'text',
                        colorscale: [
                            [0, 'rgb(167, 243, 208)'],    // Emerald-200 (good AQI)
                            [0.3, 'rgb(253, 224, 71)'],    // Yellow (moderate)
                            [0.5, 'rgb(249, 115, 22)'],   // Orange (unhealthy for sensitive)
                            [0.8, 'rgb(239, 68, 68)'],    // Red (unhealthy)
                            [1, 'rgb(185, 28, 28)']       // Dark Red (hazardous)
                        ],
                        reversescale: false,
                        marker: {
                            line: {
                                color: 'rgb(17, 24, 39)',
                                width: 0.2
                            }
                        },
                        colorbar: {
                            title: 'Median AQI',
                            thickness: 15,
                            len: 0.8,
                            x: 0.95,
                            tickcolor: '#9ca3af',
                            tickfont: { color: '#9ca3af' },
                            titlefont: { color: '#9ca3af' }
                        }
                    } as any
                ]}
                layout={{
                    geo: {
                        scope: 'usa',
                        projection: { type: 'albers usa' },
                        showlakes: true,
                        lakecolor: 'rgb(17, 24, 39)',
                        bgcolor: 'rgba(0,0,0,0)',
                        showcoastlines: false,
                        showland: true,
                        landcolor: 'rgb(31, 41, 55)',
                        subunitcolor: 'rgb(55, 65, 81)',
                    },
                    margin: { l: 0, r: 0, t: 0, b: 0 },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    autosize: true,
                    dragmode: false
                }}
                config={{
                    responsive: true,
                    displayModeBar: false,
                    scrollZoom: false
                }}
                className="w-full h-full"
                style={{ width: '100%', height: '100%' }}
            />
        </div>
    );
}
