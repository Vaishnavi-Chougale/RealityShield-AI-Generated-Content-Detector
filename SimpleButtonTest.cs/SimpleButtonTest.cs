using UnityEngine;

public class SimpleButtonTest : MonoBehaviour
{
    public GameObject handwashCube;

    public void ShowCube()
    {
        handwashCube.SetActive(true);
    }
}