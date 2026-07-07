/**
 * MuseTrack AI Dashboard Service API
 */

export const fetchCameraData = async () => {
  const response = await fetch('/api/data');
  if (!response.ok) {
    throw new Error(`Failed to fetch camera data: ${response.statusText}`);
  }
  return response.json();
};
